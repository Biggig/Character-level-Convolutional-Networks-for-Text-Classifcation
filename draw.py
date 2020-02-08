import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from model import CNN
from data.dataset import Ag_news
from config.config import get_args
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import interpolate
import matplotlib.pyplot as plt


def predict(model, opt, device, test_dataset):  # 返回准确率
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


if __name__ == "__main__":
    opt = get_args()
    model = CNN(n_classes=4, input_length=opt.maxlen, input_dim=len(opt.alphabet)+1,
                n_conv_filters=256, n_fc_neurons=1024)
    #设备
    device = torch.device("cuda:{}".format(opt.gpuid)
                          if opt.gpuid >= 0 else "cpu")
    train_accuracy = []
    train_loss = []
    test_accuracy = []
    test_loss = []
    train_dataset = Ag_news(opt=opt, train=True, test=False)
    test_dataset = Ag_news(opt=opt, train=False, test=True)
    for i in range(1, opt.epochs + 1):
        model = torch.load(opt.model_folder + opt.model_name + str(i))
        print("Succeed load model:" + opt.model_name + str(i))
        cur_accuracy, cur_loss = predict(
                    model, opt, device, train_dataset)
        train_accuracy.append(cur_accuracy)
        train_loss.append(cur_loss)
        print("train_accuracy:" + str(cur_accuracy))
        print("train_loss:" + str(cur_loss))

        accuracy, loss = predict(model, opt, device, test_dataset)
        test_accuracy.append(accuracy)
        test_loss.append(loss)
        print("test_accuracy:" + str(accuracy))
        print("test_loss:" + str(loss))

    epochs = np.linspace(1, opt.epochs + 1, 50)
    full_epochs = np.linspace(1, opt.epochs + 1, 50)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, test_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, 'bo', label='Training acc')
    plt.plot(epochs, test_accuracy, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

