"""This file trained LightGCN model with BPR loss as its loss function.
It ignored the rating, and consider its implicit information.

The nagetive sampling procedure are difficult to implement with horovod
so we did not provide the distributed version of this file.

If you are write LightGCN parameters into Neo4j database, do not forget
to start Neo4j."""

import argparse
from argparse import Namespace
from typing import Literal
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.nn import LightGCN
from torch_geometric.utils import degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_config() -> Namespace:
    parser = argparse.ArgumentParser()
    """Configurations for MovieLens 100k."""
    parser.add_argument("--num-users", type=int, default=943)
    parser.add_argument("--num-movies", type=int, default=1682)
    parser.add_argument("--train-data", default="dataset/ml-100k/u1.base")
    parser.add_argument("--test-data", default="dataset/ml-100k/u1.test")
    parser.add_argument("--full-data", default="dataset/ml-1m/u.data")

    """Configurations for MovieLens 1M."""
    # parser.add_argument("--num-users", type=int, default=6040)
    # parser.add_argument("--num-movies", type=int, default=3952)
    # parser.add_argument("--full-data", default="dataset/ml-1m/ratings.dat")

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--model", default="model.pth")
    parser.add_argument("--loss", default="loss.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--neo4j-host", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j-username", default="neo4j")
    parser.add_argument("--neo4j-passwd", default="20214919")
    args = parser.parse_args()
    return args


def load_data(mode: Literal["full", "splited"]):
    """Load data from files."""
    if mode == "full":
        data = pd.read_csv(args.full_data, sep="\t", header=None, usecols=[0, 1])
        data.columns = ["user_index", "movie_index"]
        data.loc[:, ["user_index"]] = data.loc[:, ["user_index"]] - 1
        data.loc[:, ["movie_index"]] = data.loc[:, ["movie_index"]] - 1 + args.num_users

        training_data, test_data = train_test_split(data, test_size=0.2, train_size=0.8)
    elif mode == "splited":
        training_data = pd.read_csv(args.train_data, sep="\t", header=None, usecols=[0, 1], dtype=int, engine="python")
        training_data.columns = ["user_index", "movie_index"]
        training_data["user_index"] = training_data["user_index"] - 1
        training_data["movie_index"] = training_data["movie_index"] - 1 + args.num_users

        test_data = pd.read_csv(args.test_data, sep="\t", header=None, usecols=[0, 1], dtype=int)
        test_data.columns = ["user_index", "movie_index"]
        test_data["user_index"] = test_data["user_index"] - 1
        test_data["movie_index"] = test_data["movie_index"] - 1 + args.num_users

    edge_label_index = torch.from_numpy(test_data.to_numpy()).T.long().to(device)
    edge_index = torch.from_numpy(training_data.to_numpy()).T.long().to(device)
    edge_index = torch.concatenate((edge_index, reversed(edge_index)), dim=-1)
    return edge_index, edge_label_index


def train():
    """Train the model by nagetive sampling."""
    total_loss = total_examples = 0

    for index in train_loader:
        # Sample positive and negative labels.
        pos_edge_label_index = train_edge_label_index[:, index]
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(args.num_users, args.num_users + args.num_movies,
                          (index.numel(), ), device=device)
        ], dim=0)
        edge_label_index = torch.cat([
            pos_edge_label_index,
            neg_edge_label_index,
        ], dim=1)

        optimizer.zero_grad()
        pos_rank, neg_rank = model(edge_index, edge_label_index).chunk(2)

        loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(k: int):
    """Evaluation with precision@k and recall@k."""
    emb = model.get_embedding(edge_index)
    user_emb, movie_emb = emb[:args.num_users], emb[args.num_users:]

    precision = recall = total_examples = 0
    for start in range(0, args.num_users, args.batch_size):
        end = start + args.batch_size
        logits = user_emb[start:end] @ movie_emb.t()

        """Exclude training edges."""
        mask = ((train_edge_label_index[0] >= start) &
                (train_edge_label_index[0] < end))
        logits[train_edge_label_index[0, mask] - start,
               train_edge_label_index[1, mask] - args.num_users] = float('-inf')

        """Computing precision and recall."""
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((edge_label_index[0] >= start) &
                (edge_label_index[0] < end))
        ground_truth[edge_label_index[0, mask] - start,
                     edge_label_index[1, mask] - args.num_users] = True
        node_count = degree(edge_label_index[0, mask] - start,
                            num_nodes=logits.size(0))

        topk_index = logits.topk(k, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)

        precision += float((isin_mat.sum(dim=-1) / k).sum())
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples


def write_query(query) -> None:
    """Write a query into Neo4j database."""
    driver = GraphDatabase().driver(args.neo4j_host, auth=(args.neo4j_username, args.neo4j_passwd))
    with driver.session() as session:
        session.execute_write(lambda tx, **msg: tx.run(query, **msg))
    driver.close()


def write_movie_features():
    """Write trained movie features into Neo4j database."""
    emb = model.get_embedding(edge_index)
    movie_emb = emb[args.num_users:]
    for index, embedding in enumerate(movie_emb):
        feature = ",".join(str(v) for v in embedding.tolist())
        write_query(
            r"Match (m:Movie{MovieID:%d}) "
            r"SET m.Features='%s';" 
            % (index + 1, feature)
        )


if __name__ == "__main__":
    args = get_config()
    torch.manual_seed(0)

    num_nodes = args.num_users + args.num_movies
    edge_index, edge_label_index = load_data("splited")
    mask = edge_index[0] < edge_index[1]
    train_edge_label_index = edge_index[:, mask]

    """Un-comment the following lines to write trained movie features into Neo4j database."""
    # model = torch.load("saved_model/model-100k-topn.pth")
    # write_movie_features()
    # exit()

    train_loader = DataLoader(
        range(train_edge_label_index.size(1)),
        shuffle=True,
        batch_size=args.batch_size,
    )

    model = LightGCN(
        num_nodes=num_nodes,
        embedding_dim=256,
        num_layers=3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best = float("inf")
    for epoch in range(1, args.epochs):
        loss = train()
        precision, recall = test(k=20)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Precision@20: '
            f'{precision:.4f}, Recall@20: {recall:.4f}')
        if recall < best:
            torch.save(model, "model-100k-topn.pth")
        with open("ml-1m-1.txt", "a", encoding="utf-8") as output:
            output.write("{},{},{}\n".format(loss, precision, recall))


"""
ml-100k u1 64 3 p0.414706 r0.291163
ml-100k u2 64 3 p0.343798 r0.332949
ml-100k u3 64 3 p0.283314 r0.343928
ml-100k u4 64 3 p0.273510 r0.354372
ml-100k u5 64 3 p0.259493 r0.350043
"""