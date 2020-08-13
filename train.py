from __future__ import print_function
import os
import csv
import argparse
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
from utils import iou
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Density-Fixing PyTorch Training')
parser.add_argument("--dataset", type=str, default="voc")
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--gamma", type=float, default=0.1, help="density-fixing parameter")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
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

best_acc = 0  # best test accuracy
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

root = "../."
root_dataset = f"{root}/torch_datasets/data"

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
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)

# prior histgram
import itertools
# prior_y: torch.Size(img_size, img_size, classes)
prior_y = torch.zeros(args.img_size, args.img_size, n_classes)

# target_all: torch.Size(n_imgs, img_size, img_size)
target_all = torch.cat([target.unsqueeze(0).float() for _, target in trainset], axis=0)

for i, j in itertools.product(range(args.img_size), range(args.img_size)):
    prior_y[i, j] = torch.histc(target_all[:, i, j], bins=n_classes, min=0, max=n_classes)

prior_y /= len(target_all)
prior_y += torch.tensor(1e-9)
prior_y  = prior_y.permute(2, 0, 1) # [img_size, img_size, classes] => [classes, img_size, img_size]

def train(epoch, update=True, topk=(1,)):
    global prior_y
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0.0
    accuracies = []

    p_y = prior_y.to(device)

    for i, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)['out']

        preds = torch.sum(torch.softmax(outputs, 1), 0)
        preds = preds / inputs.size(0) # [classes, img_size, img_size]

        R = nn.KLDivLoss()(p_y.log(), preds)
        loss = criterion(outputs, targets.long()) + args.gamma * R
        # import pdb; pdb.set_trace()

        train_loss += loss.item()
        # accuracies += accuracy(outputs, targets, topk=topk)
        pred = outputs.max(1)[1].cpu()
        mean_iou = iou(pred, targets.cpu(), n_classes)
        mean_iou = np.nanmean(mean_iou)

        if update:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        progress_bar(i, len(trainloader),
                     'Loss: %.3f | IoU: %.3f%%'
                     % (train_loss/(i+1), mean_iou))

    return (train_loss/i, mean_iou)

def test(epoch, update=True, topk=(1,)):
    global best_acc
    net.eval()
    test_loss = 0
    accuracies = []

    mean_ious = np.array([0.0 for i in range(n_classes)])
    preds_t = []
    targets_t = []
    
    for i, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)['out']
        loss = criterion(outputs, targets.long())

        test_loss += loss.item()
        # accuracies += accuracy(outputs, targets, topk=topk)
        # acc = np.mean(accuracies)
        
        pred = outputs.max(1)[1].cpu()
        preds_t.append(pred)
        targets_t.append(targets.cpu())

        mean_iou = iou(pred, targets.cpu(), n_classes)
        mean_iou = np.nanmean(mean_iou)

        progress_bar(i, len(testloader),
                     'Loss: %.3f | IoU: %.3f%%'
                     % (test_loss/(i+1), mean_iou))

    ious = iou(torch.cat(preds_t), torch.cat(targets_t), n_classes=n_classes)
    acc = np.nanmean(ious)

    if update:
        if acc > best_acc:
            checkpoint(acc, epoch)
            best_acc = acc

    # return (test_loss/i, np.mean(accuracies))
    return (test_loss/i, acc)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, f'{root}/checkpoint/{args.dataset}/ckpt.t7' + args.name + '_'
               + str(args.seed))


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
        logwriter.writerow(['epoch', 'train loss', 'train acc',
                            'test loss', 'test acc'])

if not args.test:
    for epoch in range(start_epoch, args.n_epochs):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        adjust_learning_rate(optimizer, epoch)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, test_loss,
                                test_acc])
else:
    for k in [1, 5]:
        test_loss, test_acc = test(1, update=False, topk=(k,))
        train_loss, train_acc = train(1, update=False, topk=(k,))
        print("Top{} Train Acc={}".format(k, train_acc))
        print("Top{} Test Acc={}".format(k, test_acc))

    print("train_loss=", train_loss)
    print("test_loss=", test_loss)
    print("diff=", test_loss - train_loss)
