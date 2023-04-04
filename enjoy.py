import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.models import resnet50

from train import convert_type, get_val_transform, rename_files, val_one_epoch
from train import DIR_TEST
from train import SmokingDataset


def rename_data():
    dir_src_path = os.path.join(DIR_TEST, "smoking")
    rename_files(dir_src_path, DIR_TEST)
    dir_src_path = os.path.join(DIR_TEST, "not_smoking")
    rename_files(dir_src_path, DIR_TEST)
    convert_type(DIR_TEST)


if __name__ == "__main__":
    # only once
    # rename_data()

    images = os.listdir(DIR_TEST)
    smoking_list = [image for image in images if image.split(".")[0] == "smoking"]
    not_smoking_list = [image for image in images if image.split(".")[0] == "not_smoking"]

    print("No. of smoking Images: {}".format(len(smoking_list)))
    print("No. of not smoking Images: {}".format(len(not_smoking_list)))

    class_to_int = {"smoking": 0, "not_smoking": 1}
    int_to_class = {0: "smoking", 1: "not_smoking"}

    test_dataset = SmokingDataset(images=images, class_to_int=class_to_int, directory=DIR_TEST, mode="train", transforms=get_val_transform())

    batch_size = 16
    num_worker = 4

    test_data_loader = DataLoader(
        dataset=test_dataset,
        num_workers=num_worker,
        batch_size=batch_size,
        shuffle=True,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = resnet50()
    model.fc = nn.Sequential(
        # nn.Sequential(*(list(model.children())[:-2])),
        # nn.AvgPool2d(kernel_size=(7, 7)),
        # nn.Flatten(),
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
        # nn.Softmax(),
        nn.Sigmoid(),
    )
    state_dict = torch.load("resnet50_best.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    criterion = nn.BCELoss()

    val_logs = {"loss": [], "accuracy": [], "time": []}
    best_val_acc = 0.0

    model.eval()
    loss, acc, _time, best_val_acc = val_one_epoch(test_data_loader, best_val_acc, model=model, criterion=criterion, device=device, val_logs=val_logs)

    print("")
    print("Validating")
    print("Loss : {}".format(round(loss, 4)))
    print("Acc : {}".format(round(acc, 4)))
    print("Time : {}".format(round(_time, 4)))
