from glob import glob
from os.path import join

from PIL import Image
import torch.utils.data as data
from torchvision import transforms


class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


class LoadDataset(data.Dataset):

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # path ex 'Dataset/train/0/*.png'
        img_path = self.file_list[index]
        img = Image.open(img_path).convert("RGB")

        img_transformed = self.transform(img, self.phase)
        label = int(img_path.split('/')[-2])

        return img_transformed, label


def make_datapath_list():
    train_dic = {'0': [], '1': []}
    val_dic = {'0': [], '1': []}
    test_dic = {'0': [], '1': []}

    for label in ['0', '1']:
        train_dic[label] = glob(join('Dataset', 'train', label, '*.png'))

    for label in ['0', '1']:
        val_dic[label] = glob(join('Dataset', 'val', label, '*.png'))

    for label in ['0', '1']:
        test_dic[label] = glob(join('Dataset', 'test', label, '*.png'))

    print('TRAIN SIZE (NORMAL): ', len(train_dic['0']))
    print('TRAIN SIZE (ABNORMAL): ', len(train_dic['1']))
    print('VAL SIZE (NORMAL): ', len(val_dic['0']))
    print('VAL SIZE (ABNORMAL): ', len(val_dic['1']))
    print('TEST SIZE (NORMAL): ', len(test_dic['0']))
    print('TEST SIZE (ABNORMAL): ', len(test_dic['1']))

    train_list = train_dic['0'] + train_dic['1']
    val_list = val_dic['0'] + val_dic['1']
    test_list = test_dic['0'] + test_dic['1']

    return train_list, val_list, test_list


def make_dataLoader():

    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 32

    train_list, val_list, test_list = make_datapath_list()

    train_dataset = LoadDataset(file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
    val_dataset = LoadDataset(file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')
    test_dataset = LoadDataset(file_list=test_list, transform=ImageTransform(size, mean, std), phase='val')

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {"train": train_dataloader, "val": val_dataloader, 'test': test_dataloader}
