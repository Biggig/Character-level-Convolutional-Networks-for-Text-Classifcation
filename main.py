import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from model import CNN
from sklearn import metrics
from utils.lib import get_metrics, CharVectorizer
from data.dataset import Ag_news
from config.config import get_args
import torchnet.meter as meter
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable

def train(model, opt, device):
    global train_accuracy, train_loss, test_accuracy, test_loss
    #数据
    train_dataset = Ag_news(opt=opt, train=True, test=False)
    #test_dataset = Ag_news(opt=opt, train=False, test=True)
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    #testloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    if opt.gpuid >= 0:
        model.cuda(device)
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.SGD(
        model.parameters(), lr=opt.lr, momentum=opt.momentum)
    if opt.solver == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    

    nclasses = len(list(model.parameters())[-1])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, opt.lr_halve_interval, gamma=opt.gamma, last_epoch=-1)
    
    for epoch in range(1, opt.epochs+1):
        pbar = tqdm(total=len(trainloader),
                    desc="Epoch {} - {}".format(epoch, "train"))
        cm = np.zeros((nclasses, nclasses), dtype=int)
        epoch_loss = 0
        for step, (label, txt) in enumerate(trainloader, 0):#读取了整个batch的数据，如何输入网络，且数据需要处理
            label = Variable(label)
            txt = Variable(txt)
            if opt.gpuid >= 0:
                label = label.cuda(device)
                txt = txt.cuda(device)
            optimizer.zero_grad()

            txt = txt.float()
            score = model(txt)
            ty_prob = F.softmax(score, 1)  # probabilites

            y_true = label.detach().cpu().numpy()
            y_pred = ty_prob.max(1)[1].cpu().numpy()

            cm += metrics.confusion_matrix(y_true, y_pred, labels=range(nclasses))
            dic_metrics = get_metrics(cm, 'accracy')

            loss = criterion(score, label)
            epoch_loss += loss.item()
            dic_metrics['logloss'] = epoch_loss/(step+1)

            loss.backward()
            optimizer.step()
            dic_metrics['lr'] = optimizer.state_dict()['param_groups'][0]['lr']

            pbar.update(1)
            pbar.set_postfix(dic_metrics)

        scheduler.step()

        test_dataset = Ag_news(opt=opt, train=False, test=True)
        accuracy, loss = predict(model, opt, device, test_dataset)
        print("\n Accuary for test dataset = " + str(accuracy))

        if (epoch % opt.snapshot_interval == 0) and (epoch > 0):  # 间断点
            path = "{}/model_epoch_{}".format(opt.model_folder, epoch)
            print("snapshot of model saved as {}".format(path))
            torch.save(model, path)

def predict(model, opt, device, test_dataset):#返回准确率
    model.eval()
    samples = len(test_dataset)
    trues = 0
    testloader = DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    for step, (label, txt) in enumerate(testloader):
        label = Variable(label)
        txt = Variable(txt)
        if opt.gpuid >= 0:
            label = label.cuda(device)
            txt = txt.cuda(device)

        txt = txt.float()
        score = model(txt)
        ty_prob = F.softmax(score, 1)  # probabilites

        y_true = label.detach().cpu().numpy()
        y_pred = ty_prob.max(1)[1].cpu().numpy()

        loss = criterion(score, label)
        total_loss += loss.item()

        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                trues += 1
    return trues / samples, total_loss / samples

if __name__=="__main__":
    opt = get_args()
    model = CNN(n_classes=4, input_length=opt.maxlen, input_dim=len(opt.alphabet)+1,
                    n_conv_filters=256, n_fc_neurons=1024)
    #设备
    device = torch.device("cuda:{}".format(opt.gpuid)
                          if opt.gpuid >= 0 else "cpu")
    if not opt.existed: 
        train(model, opt, device)
    else:
        model = torch.load(opt.model_folder + opt.model_name + str(5))
    
    

