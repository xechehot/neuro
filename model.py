import pandas as pd
import numpy as np
import random
import sys
import xlrd
from collections import defaultdict
import pickle
import re
import argparse
import os
import time

# for reading and displaying images
from skimage.io import imread
from pathlib import Path
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, \
    BatchNorm2d, Dropout, Sigmoid, MSELoss, L1Loss
from torch.optim import Adam, SGD, Adadelta
from torch.utils.data import DataLoader, TensorDataset, Dataset, Sampler
from torchvision.transforms import Compose, RandomCrop, RandomResizedCrop, ToPILImage, ToTensor, Lambda,\
    RandomHorizontalFlip, RandomRotation, RandomAffine, RandomPerspective, Grayscale, Normalize, Resize, CenterCrop, Pad, Normalize

# import glow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
size = (1920, 844)
cropped_size = (256, 128)
model_name = 'model'
random.seed(34)
criterion_number = 6
crit = 2

train_f = ('-t' in sys.argv)
cont_f = ('-c' in sys.argv)
files_f = ('-f' in sys.argv)


def read_xl(filename):
    wb = xlrd.open_workbook(filename)
    sheet = wb.sheet_by_index(0)
    i = 7
    lines = []
    while i < sheet.nrows:
        pic = str(int(sheet.cell_value(i, 0))) + '.png'
        marks = []
        for i in range(i+1, i+1+criterion_number):
            mark = int(sheet.cell_value(i, 1))
            marks.append(mark)
        line = (pic, *marks)
        lines.append(line)
        i += 1
    return lines


def create_files(file_path, val):
    lines = []
    lines = read_xl(file_path)
    if val:
        with open('val.pkl', 'wb') as valfile:
            pickle.dump(lines, valfile, protocol=pickle.HIGHEST_PROTOCOL)
            return

    random.shuffle(lines)
    train_size = int(0.8 * len(lines))
    train_xy = lines[:train_size]
    test_xy = lines[train_size:]
    with open('train.pkl', 'wb') as trainfile:
        pickle.dump(train_xy, trainfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open('test.pkl', 'wb') as testfile:
        pickle.dump(test_xy, testfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_last(model_name, model, root='checkpoints'):
    root = Path(root)
    pat = re.compile(r'.*_(?P<epoch>[0-9]+)_(?P<loss>[0-9\.]+).pth')

    matches = (pat.match(p.name) for p in root.glob(f'{model_name}_*.pth'))
    epoch, m = max(((int(m['epoch']), m) for m in matches), key=lambda x: x[0])

    path = root / m.string
    model.load_state_dict(torch.load(path, map_location='cpu'))
    print('Loading: ' + str(path))
    return epoch, float(m['loss'])


def load_dataset(data_file):
    image_paths = []
    scores = []
    with open(data_file, 'rb') as fp:
        XY = pickle.load(fp)
    for line in XY:
        image_paths.append(line[0])
        scores.append(line[1:])
        # scores.append(line[crit])

    Y = np.array(scores, 'float32')
    Y /= 10.0
    return image_paths, Y


class Net(Module):
    def __init__(self, init=64):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # MaxPool2d(kernel_size=2, stride=2),
            BatchNorm2d(1),
            Conv2d(1, init, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=4, stride=4),
            # Dropout(0.1),
            # Defining a 2D convolution layer
            Conv2d(init, init * 2, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.1),
            # Defining another 2D convolution layer
            Conv2d(init * 2, init * 4, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Dropout(0.1),

            Conv2d(init * 4, init * 4, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.1),
        )
        self.linear_layers = Sequential(
            Linear((cropped_size[0] // 32) * (cropped_size[1] // 32) * init * 4, init * 2),
            ReLU(inplace=True),
            Dropout(0.5),
            Linear(init * 2, 1),
            Sigmoid(),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        return self.linear_layers(x)


def load_pic(image_path):
    img = imread(image_path)
    img = img[:, :, 3]
    img = img.astype('float32')
    img /= 255.0
    img = np.array(img)
    return img


class BalancedDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.img_names = x
        resize_transform = Compose([ToPILImage(), Resize(cropped_size[::-1]), ToTensor()])
        self.data = [resize_transform(load_pic(xi)) for xi in x]
        self.xy = []
        for i in range(criterion_number):
            self.xy.append(defaultdict(list))
            for j in range(len(x)):
                yi = y[j][i]
                # self.data[int(sum(yi) > 0.5 * criterion_number)].append((resize_transform(xi), yi))
                self.xy[i][int(yi > 0.5)].append((j, yi))
        for i in range(criterion_number):
            print(len(self.xy[i][0]), len(self.xy[i][1]))
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        lable, id_, crit = idx
        pic_num, mark = self.xy[crit][lable][id_]
        mark = random.uniform(max(0.0, mark - 0.025), min(1.0, mark + 0.025))
        mark = np.array(mark).astype('float32').reshape(1)
        sample = self.data[pic_num]
        m = 0
        sample_ = sample
        if self.transform:
            while m == 0:
                sample_ = self.transform(sample)
                m = torch.max(sample_)
        # sample = torch.clamp(sample, 0.0, 1.0)
        # sample[sample > 0] = 1
        sample_ /= torch.max(sample_)
        return sample_, mark


class DrawingDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.img_names = x
        resize_transform = Compose([ToPILImage(), Resize(cropped_size[::-1]), ToTensor()])
        self.x = [resize_transform(load_pic(xi)) for xi in x]
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        id_, crit = idx
        sample = self.x[id_]
        mark = self.y[id_][crit].reshape(1)
        sample_ = sample
        if self.transform:
            sample_ = self.transform(sample)

        # sample = torch.clamp(sample, 0.0, 1.0)
        # sample[sample > 0] = 1
        sample_ = sample_ / torch.max(sample_)
        return sample_, mark

    def get_img_name(self, id_):
        return self.img_names[id_]


class BalancedSampler(Sampler):
    def __init__(self, ds: BalancedDataset, total: int, weights=(1, 1), crit=1):
        crit -= 1
        self.weights = weights
        self.lens = {label: len(record) for label, record in ds.xy[crit].items()}
        self.total = total
        self.rng = random.Random()
        self.crit = crit

    def __len__(self):
        return self.total

    def __iter__(self):
        data = sorted(self.lens.items())
        for _ in range(len(self)):
            for label, len_ in self.rng.choices(data, weights=self.weights):
                yield label, self.rng.randrange(len_), self.crit


class DrawingSampler(Sampler):
    def __init__(self, ds: DrawingDataset, crit=1):
        crit -= 1
        self.total = len(ds)
        self.crit = crit

    def __len__(self):
        return self.total

    def __iter__(self):
        for i in range(self.total):
            yield i, self.crit


def random_ext(x):
    size = x.size
    ratio_1 = random.uniform(1, 5)
    ratio_2 = random.uniform(1, 5)
    res = Resize(size=(int(size[1] * ratio_1), int(size[0] * ratio_2)))(x)
    res = RandomCrop(size=(5 * size[1], 5 * size[0]), pad_if_needed=True)(res)
    res = Resize(size=size[::-1])(res)
    return res


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, crit, model_name, epochs=25000, start_epoch=1):
    train_x, train_y = load_dataset('train.pkl')
    test_x, test_y = load_dataset('test.pkl')
    model_name = model_name + str(crit)

    batch_size = 10
    batches = 5

    transform = Compose(
        [ToPILImage(), Pad((75, 30), 0), RandomAffine(degrees=15, scale=(0.6, 1.5), shear=50), Lambda(random_ext),
         RandomCrop(cropped_size[::-1]),
         RandomHorizontalFlip(),
         ToTensor()])
    train_dataset = BalancedDataset(train_x, train_y, transform=transform)
    test_dataset = DrawingDataset(test_x, test_y)
    # plt.figure(figsize=(8, 8))
    # for i in range(8):
    #     plt.subplot(4, 2, i + 1), plt.imshow(np.array(train_dataset.__getitem__((i % 2, i, crit-1))[0][0]), cmap='binary')
    # plt.show()

    trainloader = DataLoader(train_dataset, sampler=BalancedSampler(train_dataset, batch_size * batches, weights=(1, 1),
                                                                    crit=crit), batch_size=batch_size)
    testloader = DataLoader(test_dataset, sampler=DrawingSampler(test_dataset, crit=crit), batch_size=batch_size)

    best_loss = float('+Inf')
    if cont_f:
        start_epoch, best_loss = load_last(model_name, model)
    model.to(device)

    criterion = MSELoss()
    optimizer = Adadelta(model.parameters(), lr=1.0)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)

    with tqdm(range(start_epoch, epochs), desc='Epochs: ', position=0, initial=start_epoch, total=epochs) as pbar:
        for epoch in pbar:
            model.train()
            running_loss = 0.0
            for x, y in trainloader:
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                loss_train = criterion(output, y)
                loss_train.backward()
                optimizer.step()
                running_loss += loss_train.item()
                del output, loss_train
            running_loss /= len(trainloader)
            scheduler.step()
            # pbar.write(str(get_lr(optimizer)))

            model.eval()
            running_test_loss = 0.0
            for x, y in testloader:
                with torch.no_grad():
                    x = x.to(device)
                    y = y.to(device)
                    output = model(x)
                    loss = criterion(output, y)
                running_test_loss += loss.item()
            running_test_loss /= len(testloader)

            if best_loss > running_test_loss:
                Path('checkpoints').mkdir(exist_ok=True)
                torch.save(model.state_dict(), f'checkpoints/{model_name}_{epoch:04d}_{running_test_loss:.4f}.pth')
                pbar.write(f'Saving at epoch {epoch}, test loss: {running_test_loss}')
                best_loss = running_test_loss

            pbar.set_postfix({
                'loss': f'{running_loss:.4f}',
                'test_loss': f'{running_test_loss:.4f}'
            })


def test(model, model_name_, val, predict_folder):
    set_name = 'test.pkl'
    if val:
        set_name = 'val.pkl'
    if predict_folder:
        test_x = [os.path.join(predict_folder, i) for i in os.listdir(predict_folder)]
        test_y = np.zeros((len(test_x), criterion_number))
    else:
        test_x, test_y = load_dataset(set_name)
    test_dataset = DrawingDataset(test_x, test_y)
    y_true = np.zeros((len(test_x), criterion_number))
    y_pred = np.zeros((len(test_x), criterion_number))
    model.to(device)
    for i in range(1, criterion_number + 1):
        testloader = DataLoader(test_dataset, sampler=DrawingSampler(test_dataset, crit=i), batch_size=len(test_x))
        model_name = model_name_ + str(i)
        load_last(model_name, model)
        model.eval()
        for x, y in testloader:
            x = x.to(device)
            with torch.no_grad():
                output = model(x)
            output = output.cpu()
            y_true[:, i - 1] = y.numpy().flatten()
            y_pred[:, i - 1] = output.numpy().flatten()
    if not predict_folder:
        for i in range(len(y_pred)):
            print(f'{test_dataset.get_img_name(i)}: {[int(round(j * 10)) for j in y_pred[i]]} {[int(round(j * 10)) for j in y_true[i]]}')

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        print(f'mse: {((y_true - y_pred)**2).mean(axis=(0, 1))}')
    else:
        for i in range(len(y_pred)):
            print(f'{test_dataset.get_img_name(i)}: {[int(round(j * 10)) for j in y_pred[i]]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_const', const=True, default=False)
    parser.add_argument('-c', '--continue', action='store_const', const=True, default=False)
    parser.add_argument('-v', '--validate', action='store_const', const=True, default=False)
    parser.add_argument('-f', '--files', nargs='?')
    parser.add_argument('-n', '--number', nargs='?', type=int, default=1)
    parser.add_argument('-m', '--model', nargs='?', default='model')
    parser.add_argument('-p', '--predict', nargs='?')

    namespace = parser.parse_args(sys.argv[1:])

    if namespace.files:
        # filename = 'Zapolnennoe_po_faktoram.xlsx'
        create_files(namespace.files, namespace.validate)
        exit()

    model = Net()

    if namespace.train:
        train(model, namespace.number, namespace.model)
    else:
        test(model, namespace.model, namespace.validate, namespace.predict)
