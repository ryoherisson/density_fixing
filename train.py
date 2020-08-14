from __future__ import print_function
import os
import csv
import argparse
import itertools
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import torchvision.models.segmentation as segmentation

from dataset import Compose, Resize, Normalize_Tensor
from utils import progress_bar
from utils import accuracy
from metrics.metrics import Metrics

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Density-Fixing PyTorch Training')
parser.add_argument("--dataset", type=str, default="voc")
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--gamma", type=float, default=0.0, help="density-fixing parameter")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--resume", action="store_true", default=False, help="resume from checkpoint")
parser.add_argument("--name", type=str, default="0", help="name of run")
parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--n_epochs", type=int, default=200, help="total epochs to run")
parser.add_argument("--decay", type=float, default=1e-4, help="weight decay")
args = parser.parse_args()

# check gpu
if torch.cuda.is_available():
    print("available gpu..")
    # device = torch.device("cuda:0")
    device = torch.device("cuda")
else:
    print("use cpu..")
    device = torch.device("cpu")

best_loss = 1e9  # best test loss
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# preparing data
print("==> Preparing data...")
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

root = "."
root_dataset = f"./{root}/torch_datasets/data"

if args.dataset == "voc":
    n_classes = 21
    color_mean, color_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transforms = Compose([Resize(args.img_size), Normalize_Tensor(color_mean, color_std)])
    trainset = datasets.VOCSegmentation(root=root_dataset, year="2012", image_set="train", download=True, transforms=transforms)
    testset = datasets.VOCSegmentation(root=root_dataset, year="2012", image_set="val", download=True, transforms=transforms)
else:
    raise NotImplementedError

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

if args.resume:
    # Load checkpoint
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
        + str(args.seed))

    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print("==> Building model..")
    # net = ResNet18(n_classes=n_classes)
    net = segmentation.fcn_resnet50(num_classes=n_classes)

if not os.path.isdir("results"):
    os.mkdir("results")

logname = (f'{root}/results/{args.dataset}/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

net = net.to(device)
net = nn.DataParallel(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
# optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)

# Metrics
metrics = Metrics(n_classes=n_classes)

# Prior histgram
# prior_y: torch.Size(img_size, img_size, classes)
prior_y = torch.zeros(args.img_size, args.img_size, n_classes)

# target_all: torch.Size(n_imgs, img_size, img_size)
target_all = torch.cat([target.unsqueeze(0).float() for _, target in trainset], axis=0)

for i, j in itertools.product(range(args.img_size), range(args.img_size)):
    prior_y[i, j] = torch.histc(target_all[:, i, j], bins=n_classes, min=0, max=n_classes)

prior_y /= len(target_all)
prior_y += torch.tensor(1e-9) # [img_size, img_size, classes]

def train(epoch, update=True, topk=(1,)):
    global prior_y
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0.0
    train_kldivloss = 0.0
    accuracies = []

    p_y = prior_y.to(device)

    for i, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)['out']

        preds = torch.softmax(outputs, 1)  # [batch-size, classes, img_size, img_size]
        preds = preds.permute(0, 2, 3, 1) # [batch-size, img_size, img_size, classes]

        p_y_ex = p_y.expand(preds.size(0), preds.size(1), preds.size(2), preds.size(3))
    
        R = nn.KLDivLoss()(p_y_ex.log(), preds)
        kldivloss = args.gamma * R
        loss = criterion(outputs, targets.long()) + kldivloss
        train_loss += loss.item()
        train_kldivloss += kldivloss.item()

        pred = torch.softmax(outputs, 1).max(1)[1].cpu()
        metrics.update(preds = pred.cpu().detach().clone(), 
                       targets = targets.cpu().detach().clone(), 
                       loss=loss.item(),
                       kldivloss=kldivloss.item())

        if update:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        progress_bar(i, len(trainloader),
                     'Loss: %.3f | KL Loss: %.3f'
                     % (train_loss/(i+1), train_kldivloss/(i+1)))

    loss, kldivloss, mean_iou = metrics.calc_metrics()
    metrics.initialize()

    return (loss, kldivloss, mean_iou)

def test(epoch, update=True, topk=(1,)):
    global best_loss
    global prior_y
    net.eval()
    test_loss = 0.0
    test_kldivloss = 0.0
    
    p_y = prior_y.to(device)

    for i, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)['out']
        loss = criterion(outputs, targets.long())

        preds = torch.softmax(outputs, 1)  # [batch-size, classes, img_size, img_size]
        preds = preds.permute(0, 2, 3, 1) # [batch-size, img_size, img_size, classes]

        p_y_ex = p_y.expand(preds.size(0), preds.size(1), preds.size(2), preds.size(3))

        R = nn.KLDivLoss()(p_y_ex.log(), preds)
        kldivloss = args.gamma * R
        loss = criterion(outputs, targets.long()) + kldivloss
        test_loss += loss.item()
        test_kldivloss += kldivloss.item()
        
        pred = torch.softmax(outputs, 1).max(1)[1].cpu()
        metrics.update(preds = pred.cpu().detach().clone(), 
                       targets = targets.cpu().detach().clone(), 
                       loss=loss.item(),
                       kldivloss=kldivloss.item())

        progress_bar(i, len(testloader),
                     'Loss: %.3f | KL Loss: %.3f'
                     % (test_loss/(i+1), test_kldivloss/(i+1)))

    loss, kldivloss, mean_iou = metrics.calc_metrics()
    metrics.initialize()

    if update:
        if epoch >= args.n_epochs - 12:
            print('saving checkpoint...')
            checkpoint(loss, epoch, best=False)

        if best_loss > loss:
            print('saving best checkpoint...')
            checkpoint(loss, epoch, best=True)
            best_loss = loss

    return (loss, kldivloss, mean_iou)


def checkpoint(loss, epoch, best=False):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'loss': loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    if best:
        torch.save(state, f'{root}/checkpoint/{args.dataset}/{args.name}/ckpt.t7' + args.name + '_'
                + str(args.seed) + '_best')
    else:
        torch.save(state, f'{root}/checkpoint/{args.dataset}/{args.name}/ckpt.t7' + args.name + '_'
                + str(args.seed) + f'_{epoch}epoch')



def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if not os.path.exists(logname) and not args.test:
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'train_kldivloss', 'train iou',
                            'test loss', 'test_kldivloss', 'test iou'])

if not args.test:
    for epoch in range(start_epoch, args.n_epochs):
        train_loss, train_kldivloss, train_iou = train(epoch)
        test_loss, test_kldivloss, test_iou = test(epoch)

        print(f"Train loss:{train_loss}, kldivloss:{train_kldivloss}, mean_iou:{train_iou}")
        print(f"Test loss:{test_loss}, kldivloss:{test_kldivloss}, mean_iou:{test_iou}")
        adjust_learning_rate(optimizer, epoch)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_kldivloss, train_iou, test_loss,
                                test_kldivloss, test_iou])
else:
    for k in [1, 5]:
        test_loss, test_iou = test(1, update=False, topk=(k,))
        train_loss, train_iou = train(1, update=False, topk=(k,))
        print("Top{} Train IoU={}".format(k, train_iou))
        print("Top{} Test IoU={}".format(k, test_iou))

    print("train_loss=", train_loss)
    print("test_loss=", test_loss)
    print("diff=", test_loss - train_loss)
