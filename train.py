import os
import random
from glob import glob

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import VGG16
from data_load import make_dataLoader


def train(model, dataloader_dict, criterion, optimizer, num_epochs, log_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用デバイス：", device)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    min_val_loss = 1e9
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        print(''.format(epoch+1, num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            writer.add_scalar('loss/' + phase, epoch_loss, epoch)
            writer.add_scalar('acc/' + phase, epoch_acc, epoch)

            print('Epoch {}/{}  {} Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_loss < min_val_loss:
                min_val_loss = epoch_loss
                save_path = './task2.pth'
                torch.save(model.state_dict(), save_path)
                print('model is saved')
    writer.close()


if __name__=='__main__':
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    epochs = 50
    num_log_dir = str(len(glob('logs/*')) + 1)
    log_dir = os.path.join('logs', 'log' + num_log_dir)
    os.makedirs(log_dir, exist_ok=True)

    dataloader_dict = make_dataLoader()
    vgg16 = VGG16()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg16.parameters(), lr=0.01, momentum=0.9)

    train(vgg16, dataloader_dict, criterion, optimizer, epochs, log_dir)
