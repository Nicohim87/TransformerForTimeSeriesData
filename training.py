import os
import sys
import csv
import time
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

from transformer_architecture_base import Decoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.__version__)
print(DEVICE)

log_file = "model/training_log.csv"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint",
    type=str,
    default="0",
    help="'latest' or an integer epoch (e.g. 100)"
)

args = parser.parse_args()

if args.checkpoint == "latest":
    checkpoint = "latest"
elif args.checkpoint == "0":
    checkpoint = None
else:
    try:
        checkpoint = f"epoch_{int(args.checkpoint)}"
        if not checkpoint % 100:
            raise
    except ValueError:
        parser.error("checkpoint must be 'latest' or an integer multiple of 100")

with open("parameters.json", 'r') as f:
    params = json.load(f)

# Read Data
with open("./data/entity_seq_pair.json", 'r') as f:
    pairs = json.load(f)

ds_train = pairs["train"]
ds_val = pairs["val"]
ds_test = pairs["test"]

df = pd.read_csv("./data/scaled_data.csv")
df = df.set_index(["City_id", "Seq_id"])


# Data Loader
from torch.utils.data import DataLoader

train_loader = DataLoader(ds_train, batch_size = params["batch_size"], shuffle=True)
val_loader = DataLoader(ds_val, batch_size = params["batch_size"])
test_loader = DataLoader(ds_test, batch_size = params["batch_size"])

# Model Preparation
from transformers import get_cosine_schedule_with_warmup

col_count = len(df.columns)

model = Decoder(
    hidden_dim=params["hidden_dim"],
    seq_len=params["seq_len"],
    corpus_size=col_count - 4,
    n_heads=params["n_attn_heads"],
    n_blocks=params["n_blocks"],
    dropout=params["dropout"],
    use_embedding=False,
    embedding_replacement=torch.nn.Linear(col_count, params["hidden_dim"])
).to(DEVICE)

if checkpoint:
    try:
        model.load_state_dict(torch.load(f"model/{checkpoint}.pth", map_location="cpu", weights_only=True))
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint at {checkpoint} epochs not found")
    model.to(DEVICE)
    log = pd.read_csv("model/training_log.csv")
    epoch_start = int(log["epoch"].max()) + 1
else:
    epoch_start = 0
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_rmse",
            "train_time",
            "val_loss",
            "val_rmse",
            "val_time",
            "lr",
        ])

criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(params["epochs"]*len(train_loader)*0.1),
    num_training_steps=int(params["epochs"]*len(train_loader)*1.1)
)

print("Device:", next(model.parameters()).device)
print("Parameter Count:", f"{sum(p.numel() for p in model.parameters()):,}")
print("Allocated Memory:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("Reserved memory:", torch.cuda.memory_reserved() / 1024**3, "GB")


# ---- Training Prep ----
def load_tensor(entities, seq):
    regional_data = [] 

    for entity, sid in zip(entities, seq):
        sid = int(sid)
        entity = entity.item() if hasattr(entity, "item") else entity

        regional_data.append(torch.tensor(df.loc[entity].loc[sid:sid + params["seq_len"]].to_numpy()))


    regional_data = torch.stack(regional_data).to(DEVICE, torch.float32)
    return regional_data

train_metrics = {
    "epoch": [],
    "train_loss": [],
    "train_rmse": [],
    "train_time":[],
    "val_loss": [],
    "val_rmse": [],
    "val_time":[],
    "lr":[],
}


# ---- Training Loop ----
for e in tqdm(
    range(epoch_start, params["epochs"]),
    desc="Epoch",
    initial=epoch_start,
    total=params["epochs"],
):
    model.train()
    train_loss = 0
    train_se = 0
    train_count = 0
    train_time = time.time()
    for entities, years in tqdm(train_loader, desc=f"    Training", leave=False):
        optimizer.zero_grad()
        regional_data = load_tensor(entities, years)
        pred = model(regional_data[:, :-1])
        ground_truth = regional_data[:, 1:, 4:]
    

        loss = criterion(pred, ground_truth)
        train_loss += loss.item()

        train_se += ((pred - ground_truth)**2).sum().item()
        train_count += pred.shape[0]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    train_metrics["train_time"].append(time.time() - train_time)
    train_metrics["train_loss"].append(train_loss/len(train_loader))
    train_metrics["train_rmse"].append((train_se/(train_count*col_count*params["seq_len"]))**0.5)


    model.eval()
    val_loss = 0
    val_se = 0
    val_count = 0
    val_time = time.time()
    with torch.no_grad():
        for entities, years in tqdm(val_loader, desc=f"    Val", leave=False):
            regional_data = load_tensor(entities, years)
            pred = model(regional_data[:, :-1])
            ground_truth = regional_data[:, -1, 4:]
            pred = pred[:, -1]
        

            loss = criterion(pred, ground_truth)
            val_loss += loss.item()

            val_se += ((pred - ground_truth)**2).sum().item()
            val_count += pred.shape[0]

    train_metrics["val_time"].append(time.time()-val_time)
    train_metrics["val_loss"].append(val_loss/len(val_loader))
    train_metrics["val_rmse"].append((val_se/(val_count*col_count))**0.5)

    train_metrics["lr"].append(scheduler.get_last_lr()[0])
    train_metrics["epoch"].append(e)

    if e%100 == 0 and e != 0:
        torch.save(model.state_dict(), f"model/epoch_{e}.pth")

    if e%5 == 0 and e != 0:
        torch.save(model.state_dict(), f"model/latest.pth")
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            for i in range(len(train_metrics["epoch"])):
                writer.writerow([
                    train_metrics["epoch"][i],
                    train_metrics["train_loss"][i],
                    train_metrics["train_rmse"][i],
                    train_metrics["train_time"][i],
                    train_metrics["val_loss"][i],
                    train_metrics["val_rmse"][i],
                    train_metrics["val_time"][i],
                    train_metrics["lr"][i],
                ])

            train_metrics["train_loss"] = []
            train_metrics["train_rmse"] = []
            train_metrics["train_time"] = []
            train_metrics["val_loss"] = []
            train_metrics["val_rmse"] = []
            train_metrics["val_time"] = []
            train_metrics["lr"] = []
            train_metrics["epoch"] = []

torch.save(model.state_dict(), f"model/latest.pth")
with open(log_file, "a", newline="") as f:
    writer = csv.writer(f)
    for i in range(len(train_metrics["epoch"])):
        writer.writerow([
            train_metrics["epoch"][i],
            train_metrics["train_loss"][i],
            train_metrics["train_rmse"][i],
            train_metrics["train_time"][i],
            train_metrics["val_loss"][i],
            train_metrics["val_rmse"][i],
            train_metrics["val_time"][i],
            train_metrics["lr"][i],
        ])

    train_metrics["train_loss"] = []
    train_metrics["train_rmse"] = []
    train_metrics["train_time"] = []
    train_metrics["val_loss"] = []
    train_metrics["val_rmse"] = []
    train_metrics["val_time"] = []
    train_metrics["lr"] = []
    train_metrics["epoch"] = []