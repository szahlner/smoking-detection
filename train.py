import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.models import resnet50

from sklearn.model_selection import train_test_split

from PIL import Image

import matplotlib.pyplot as plt


DIR_TRAIN = os.path.join("data", "train")
DIR_TEST = os.path.join("data", "test")


def rename_files(dir_src_path, dir_dst_path=None, prefix=None):
    if dir_dst_path is None:
        dir_dst_path = dir_src_path
    
    if prefix is None:
        prefix = os.path.basename(dir_src_path)

    for file in os.listdir(dir_src_path):
        src_file = os.path.join(dir_src_path, file)
        if os.path.isfile(src_file):
            dst_file = os.path.join(dir_dst_path, "{}.{}".format(prefix, file))
            shutil.copyfile(src_file, dst_file)


def convert_type(dir_src_path):
    for file in os.listdir(dir_src_path):
        src_file = os.path.join(dir_src_path, file)
        if os.path.isfile(src_file) and not src_file.endswith("jpg"):
            image = Image.open(src_file)
            rgb_iamge = image.convert("RGB")
            src_file = src_file.split(".")
            src_file = ".".join(src_file[:-1])
            src_file = "{}.{}".format(src_file, "jpg")
            rgb_iamge.save(src_file)
    
    for file in os.listdir(dir_src_path):
        src_file = os.path.join(dir_src_path, file)
        if os.path.isfile(src_file) and not src_file.endswith("jpg"):
            os.remove(src_file)


def rename_data():
    dir_src_path = os.path.join(DIR_TRAIN, "smoking")
    rename_files(dir_src_path, DIR_TRAIN)
    dir_src_path = os.path.join(DIR_TRAIN, "not_smoking")
    rename_files(dir_src_path, DIR_TRAIN)
    convert_type(DIR_TRAIN)


def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomCrop(204),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class SmokingDataset(Dataset):
    def __init__(self, images, class_to_int, directory, mode="train", transforms=None):
        super(Dataset, self).__init__()
        self.images = images
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms
        self.directory = directory

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.directory, image_name)
        
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = image.convert("RGB")

        if self.mode == "train" or self.mode == "val":
            label = self.class_to_int[image_name.split(".")[0]]
            label = torch.tensor(label, dtype=torch.float32)

            image = self.transforms(image)
            return image, label

        elif self.mode == "test":
            image = self.transforms(image)
            return image

    def __len__(self):
        return len(self.images)


def accuracy(preds, trues):
    preds = [1 if preds[n] >= 0.5 else 0 for n in range(len(preds))]
    acc = [1 if preds[n] == trues[n] else 0 for n in range(len(preds))]
    
    acc = np.sum(acc) / len(preds)

    return acc * 100
   

def train_one_epoch(train_data_loader):
    epoch_loss, epoch_acc = [], []
    start_time = time.time()

    for images, labels in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1))

        optimizer.zero_grad()

        preds = model(images)
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)

        acc = accuracy(preds, labels)
        epoch_acc.append(acc)

        _loss.backward()
        optimizer.step()

    end_time = time.time()
    total_time = end_time - start_time

    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)

    train_logs["loss"].append(epoch_loss)
    train_logs["accuracy"].append(epoch_acc)
    train_logs["time"].append(total_time)

    return epoch_loss, epoch_acc, total_time


def val_one_epoch(val_data_loader, best_val_acc, model, criterion, device, val_logs):
    epoch_loss, epoch_acc = [], []
    start_time = time.time()

    for images, labels in val_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1))

        preds = model(images)
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)

        acc = accuracy(preds, labels)
        epoch_acc.append(acc)

    end_time = time.time()
    total_time = end_time - start_time

    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)

    val_logs["loss"].append(epoch_loss)
    val_logs["accuracy"].append(epoch_acc)
    val_logs["time"].append(total_time)

    if epoch_acc > best_val_acc:
        best_val_acc = epoch_acc
        torch.save(model.state_dict(), "resnet50_best.pth")

    return epoch_loss, epoch_acc, total_time, best_val_acc


if __name__ == "__main__":
    # only once
    # rename_data()

    images = os.listdir(DIR_TRAIN)
    smoking_list = [image for image in images if image.split(".")[0] == "smoking"]
    not_smoking_list = [image for image in images if image.split(".")[0] == "not_smoking"]

    print("No. of smoking Images: {}".format(len(smoking_list)))
    print("No. of not smoking Images: {}".format(len(not_smoking_list)))

    class_to_int = {"smoking": 0, "not_smoking": 1}
    int_to_class = {0: "smoking", 1: "not_smoking"}

    train_images, val_images = train_test_split(images, test_size=0.2)

    train_dataset = SmokingDataset(images=train_images, class_to_int=class_to_int, directory=DIR_TRAIN, mode="train", transforms=get_train_transform())
    val_dataset = SmokingDataset(val_images, class_to_int, directory=DIR_TRAIN, mode="val", transforms=get_val_transform())

    batch_size = 16
    num_worker = 4

    train_data_loader = DataLoader(
        dataset=train_dataset,
        num_workers=num_worker,
        batch_size=batch_size,
        shuffle=True,
    )

    val_data_loader = DataLoader(
        dataset=val_dataset,
        num_workers=num_worker,
        batch_size=batch_size,
        shuffle=True,
    )

    for imgs, lbls in train_data_loader:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(imgs, 4).permute(1, 2, 0))
        break

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    criterion = nn.BCELoss()

    train_logs = {"loss": [], "accuracy": [], "time": []}
    val_logs = {"loss": [], "accuracy": [], "time": []}

    model.to(device)

    epochs = 100

    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        loss, acc, _time = train_one_epoch(train_data_loader)

        print("")
        print("Training")
        print("Epoch {}".format(epoch + 1))
        print("Loss : {}".format(round(loss, 4)))
        print("Acc : {}".format(round(acc, 4)))
        print("Time : {}".format(round(_time, 4)))

        model.eval()
        loss, acc, _time, best_val_acc = val_one_epoch(val_data_loader, best_val_acc, model=model, criterion=criterion, device=device, val_logs=val_logs)

        print("")
        print("Validating")
        print("Epoch {}".format(epoch + 1))
        print("Loss : {}".format(round(loss, 4)))
        print("Acc : {}".format(round(acc, 4)))
        print("Time : {}".format(round(_time, 4)))

    #Loss
    plt.figure()
    plt.title("Loss")
    plt.plot(np.arange(1, len(train_logs["loss"]) + 1), train_logs["loss"], color="blue")
    plt.plot(np.arange(1, len(val_logs["loss"]) + 1), val_logs["loss"], color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.figure()
    plt.title("Accuracy")
    plt.plot(np.arange(1, len(train_logs["accuracy"]) + 1), train_logs["accuracy"], color="blue")
    plt.plot(np.arange(1, len(val_logs["accuracy"]) + 1), val_logs["accuracy"], color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
