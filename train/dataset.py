"""
Adapted from https://github.com/usef-kh/fer/tree/master/data
"""

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from imblearn.under_sampling import RandomUnderSampler
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False):
        self.images = images
        self.labels = labels
        self.transform = transform

        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).type(torch.long)
        sample = (img, label)

        return sample


def load_data(path="./datasets/fer2013/fer2013.csv"):
    fer2013 = pd.read_csv(path)

    new_emotion_mapping = {
        0: 0, # Angry -> BAD
        1: 0, # Disgust -> BAD
        2: 0, # Fear -> BAD
        3: 1, # Happy -> GOOD
        4: 0, # Sad -> BAD
        5: 1, # Surprise -> GOOD
        6: 2, # Neutral -> NEUTRAL
    }

    return fer2013, new_emotion_mapping


def new_labels(emotions):
    mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 2}  # BAD  # GOOD  # NEUTRAL
    return [mapping[num] for num in emotions]


def prepare_data(data):
    """Prepare data for modeling
    input: data frame with labels und pixel data
    output: image and label array"""

    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data["emotion"])))
    image_label = new_labels(image_label)

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, "pixels"], dtype=int, sep=" ")
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


def get_dataloaders(path="./datasets/fer2013/fer2013.csv", bs=64, augment=True):
    """Prepare train, val, & test dataloaders
    Augment training data using:
        - cropping
        - shifting (vertical/horizental)
        - horizental flipping
        - rotation
    input: path to fer2013 csv file
    output: (Dataloader, Dataloader, Dataloader)"""

    fer2013, _ = load_data(path)

    xtrain, ytrain = prepare_data(fer2013[fer2013["Usage"] == "Training"])
    xval, yval = prepare_data(fer2013[fer2013["Usage"] == "PrivateTest"])
    xtest, ytest = prepare_data(fer2013[fer2013["Usage"] == "PublicTest"])

    undersampler = RandomUnderSampler(sampling_strategy="auto", random_state=42)

    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xval = xval.reshape(xval.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    # Fit and transform the data
    xtrain, ytrain = undersampler.fit_resample(xtrain, ytrain)
    xval, yval = undersampler.fit_resample(xval, yval)
    xtest, ytest = undersampler.fit_resample(xtest, ytest)

    # reshaping X back to the first dims
    xtrain = xtrain.reshape(-1, 48, 48)
    xval = xval.reshape(-1, 48, 48)
    xtest = xtest.reshape(-1, 48, 48)

    mu, st = 0, 255

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.TenCrop(40),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops]
                )
            ),
            transforms.Lambda(
                lambda tensors: torch.stack(
                    [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors]
                )
            ),
        ]
    )
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.5, contrast=0.5, saturation=0.5
                        )
                    ],
                    p=0.5,
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
                transforms.FiveCrop(40),
                transforms.Lambda(
                    lambda crops: torch.stack(
                        [transforms.ToTensor()(crop) for crop in crops]
                    )
                ),
                transforms.Lambda(
                    lambda tensors: torch.stack(
                        [
                            transforms.Normalize(mean=(mu,), std=(st,))(t)
                            for t in tensors
                        ]
                    )
                ),
                transforms.Lambda(
                    lambda tensors: torch.stack(
                        [transforms.RandomErasing()(t) for t in tensors]
                    )
                ),
            ]
        )
    else:
        train_transform = test_transform

    train = CustomDataset(xtrain, ytrain, train_transform)
    val = CustomDataset(xval, yval, test_transform)
    test = CustomDataset(xtest, ytest, test_transform)

    trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=2)
    valloader = DataLoader(val, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(test, batch_size=64, shuffle=True, num_workers=2)

    return trainloader, valloader, testloader
