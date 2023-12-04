"""This file trained LightGCN model with RMSE as its loss function."""

import argparse
from argparse import Namespace
from typing import Literal

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from torch_geometric.nn.models import LightGCN


def get_config() -> Namespace:
    parser = argparse.ArgumentParser()
    """Configurations for MovieLens 100k."""
    parser.add_argument("--num-users", type=int, default=943)
    parser.add_argument("--num-movies", type=int, default=1682)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-data", default="dataset/ml-100k/u1.base")
    parser.add_argument("--test-data", default="dataset/ml-100k/u1.test")
    parser.add_argument("--full-data", default="dataset/ml-100k/u.data")

    """Configuration for MovieLens 1M."""
    # parser.add_argument("--num-users", type=int, default=6040)
    # parser.add_argument("--num-movies", type=int, default=3952)
    # parser.add_argument("--embedding-dim", type=int, default=512)
    # parser.add_argument("--num-layers", type=int, default=1)
    # parser.add_argument("--epochs", type=int, default=25)
    # parser.add_argument("--full-data", default="dataset/ml-1m/ratings.dat")

    """Configurations for MovieLens 10M."""
    # parser.add_argument("--num-users", type=int, default=71567)
    # parser.add_argument("--num-movies", type=int, default=65133)
    # parser.add_argument("--embedding-dim", type=int, default=512)
    # parser.add_argument("--num-layers", type=int, default=2)
    # parser.add_argument("--full-data", default="dataset/ml-10M100K/ratings.dat")
    # parser.add_argument("--epochs", type=int, default=12)

    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--model", default="model.pth")
    parser.add_argument("--loss", default="loss.pth")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return args


class MyModel(nn.Module):
    def __init__(self, base) -> None:
        super().__init__()
        self.base = base

    def forward(self, x):
        return self.base(x.T)


class MyDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.feature = (
            torch.from_numpy(
                data.loc[:, ["user_index", "movie_index"]]
                .to_numpy()
            )
            .long()
            .to(device)
        )

        self.target = (
            torch.from_numpy(
                data.loc[:, ["rating"]]
                .to_numpy()
            )
            .reshape(-1)
            .float()
            .to(device)
        )

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.feature[index], self.target[index]


def load_data_from_csv(mode: Literal["full", "splited"]):
    if mode == "full":
        data = pd.read_csv(args.full_data, sep="\t", header=None, usecols=[0, 1, 2])
        data.columns = ["user_index", "movie_index", "rating"]
        data.loc[:, ["user_index"]] = data.loc[:, ["user_index"]] - 1
        data.loc[:, ["movie_index"]] = data.loc[:, ["movie_index"]] - 1 + args.num_users

        training_data, test_data = train_test_split(data, test_size=0.2, train_size=0.8)
    elif mode == "splited":
        training_data = pd.read_csv(args.train_data, sep="\t", header=None, usecols=[0, 1, 2])
        training_data.columns = ["user_index", "movie_index", "rating"]
        training_data.loc[:, ["user_index"]] = training_data.loc[:, ["user_index"]] - 1
        training_data.loc[:, ["movie_index"]] = training_data.loc[:, ["movie_index"]] - 1 + args.num_users

        test_data = pd.read_csv(args.test_data, sep="\t", header=None, usecols=[0, 1, 2])
        test_data.columns = ["user_index", "movie_index", "rating"]
        test_data.loc[:, ["user_index"]] = test_data.loc[:, ["user_index"]] - 1
        test_data.loc[:, ["movie_index"]] = test_data.loc[:, ["movie_index"]] - 1 + args.num_users
    else:
        raise ValueError("Invalid mode to load dataset.")
    return training_data, test_data


def get_model_and_data_loader(mode: Literal["full", "splited"]):
    num_nodes = args.num_users + args.num_movies
    base = LightGCN(num_nodes, args.embedding_dim, args.num_layers).to(device)
    model = MyModel(base)

    training_data, test_data = load_data_from_csv(mode)
    trainingset = MyDataset(training_data)
    testset = MyDataset(test_data)
    training_loader = DataLoader(trainingset, args.batch_size, True)
    test_loader = DataLoader(testset, args.batch_size, True)

    return model, training_loader, test_loader


def train_rmse(model, data_loader, loss_fn, optimizer):
    model.train()
    for X, y in data_loader:
        pred = model(X)
        loss = torch.sqrt(loss_fn(pred, y))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


@torch.no_grad()
def test_rmse(model, data_loader):
    model.eval()
    y_pred = []
    y_true = []
    for X, y in data_loader:
        pred = model(X)
        y_pred += pred.tolist()
        y_true += y.tolist()
    report = np.sqrt(mean_squared_error(y_true, y_pred))
    return report


if __name__ == "__main__":
    args = get_config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, training_loader, test_loader = get_model_and_data_loader("full")
    loss_fn = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters())
    best = np.inf

    for epoch in range(args.epochs):
        train_rmse(model, training_loader, loss_fn, optimizer)
        rmse = test_rmse(model, test_loader)
        if rmse < best:
            torch.save(model, args.model)
            best = rmse
        print("Epoch {} ends. Test loss: {}".format(epoch, rmse))
