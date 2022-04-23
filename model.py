from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models


def VGG16():
    vgg16 = models.vgg16()
    vgg16.classifier = nn.Sequential(OrderedDict([
        ("0", nn.Linear(25088, 4096)),
        ("1", nn.ReLU()),
        ("2", nn.Dropout(p=0.5)),
        ("3", nn.Linear(4096, 2))
    ]))

    return vgg16

def load_trained_model(model, model_path):
    model_weights = torch.load(model_path, map_location={'cuda': 'cpu'})
    new_state_dict = OrderedDict()

    for k, v in model_weights.items():
        k = k[7:]
        new_state_dict[k]=v

    model.load_state_dict(new_state_dict)

    return model