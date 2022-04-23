import os
import random

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import transforms

from data_load import make_dataLoader
from model import VGG16, load_trained_model


def save_cm_image(trues, preds, class_label, img_path):
    cm = confusion_matrix(trues, preds)
    cm = pd.DataFrame(data=cm, index=class_label, columns=class_label)
    sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues')
    plt.yticks(rotation=0)
    plt.xlabel("prediction", fontsize=13, rotation=0)
    plt.ylabel("trues", fontsize=13)
    plt.savefig(img_path)

def unnorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def save_images(imgs_tensor, true_label, save_dir):
    for i,(img_tensor , label) in enumerate(zip(imgs_tensor, true_label)):
        img_numpy = unnorm(img_tensor).permute(1, 2, 0).numpy().copy()
        img_numpy = (img_numpy * 255).astype(np.uint8)
        img = Image.fromarray(img_numpy)
        img.save(save_dir + str(i) + '_true_label_is_'+ str(label) + '.png')

def test(model, dataloader_dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用デバイス：", device)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    model.eval()

    correct = 0
    preds, trues, correct_imgs, wrong_imgs, correct_true_label, wrong_true_label= [], [], [], [], [], []
    for inputs, labels in tqdm(dataloader_dict["test"]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, outputs = torch.max(outputs, 1)
        correct += torch.sum(outputs == labels.data)

        c = inputs[outputs == labels]
        w = inputs[outputs != labels]
        c_l = labels[outputs == labels]
        w_l = labels[outputs != labels]

        correct_imgs.append(c.cpu().detach())
        wrong_imgs.append(w.cpu().detach())
        correct_true_label.append(c_l.cpu().detach().numpy().copy())
        wrong_true_label.append(w_l.cpu().detach().numpy().copy())
        preds.append(outputs.cpu().detach().numpy().copy())
        trues.append(labels.cpu().detach().numpy().copy())

    acc = correct.double() / len(dataloader_dict["test"].dataset)
    print("ACC : ", acc.item())

    trues = np.concatenate(trues)
    preds = np.concatenate(preds)
    cm_image_path = 'images/confusion_matrix.png'
    save_cm_image(trues, preds, ['normal', 'abnormal'], cm_image_path)

    correct_true_label = np.concatenate(correct_true_label)
    wrong_true_label = np.concatenate(wrong_true_label)

    correct_imgs_dir = 'images/correct/'
    wrong_imgs_dir = 'images/wrong/'
    os.makedirs(correct_imgs_dir, exist_ok=True)
    os.makedirs(wrong_imgs_dir, exist_ok=True)
    correct_imgs = torch.cat(correct_imgs, dim=0)
    wrong_imgs = torch.cat(wrong_imgs, dim=0)

    save_images(correct_imgs, correct_true_label, correct_imgs_dir)
    save_images(wrong_imgs, wrong_true_label, wrong_imgs_dir)

if __name__== '__main__':
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    model_path = 'task2.pth'

    dataloader_dict = make_dataLoader()
    vgg16 = VGG16()
    vgg16 = load_trained_model(vgg16, model_path)

    test(vgg16, dataloader_dict)