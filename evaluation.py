import os
import sys
import json
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformer_architecture_base import Decoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.__version__)
print(DEVICE)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint",
    type=str,
    default="latest",
    help="'latest' or an integer epoch (e.g. 100)"
)

args = parser.parse_args()

if args.checkpoint == "latest":
    checkpoint = "latest"
elif args.checkpoint == "0":
    parser.error("Checkpoint cannot be from 0. Use 'latest' or an integer multiple of 100")
else:
    try:
        checkpoint = f"epoch_{int(args.checkpoint)}"
        if not checkpoint % 100:
            raise
    except ValueError:
        parser.error("checkpoint must be 'latest' or an integer multiple of 100")

with open("parameters.json", 'r') as f:
    params = json.load(f)


# Read test dataset
with open("./data/entity_seq_pair.json", 'r') as f:
    ds = json.load(f)
ds_test = ds["test"]
SEQ_LEN = ds["metadata"]["seq_len"]
SEQ_PRED = [x + SEQ_LEN for x in ds["metadata"]["seq_starts"]]

df = pd.read_csv("./data/scaled_test_data.csv")
df = df.set_index(ds["metadata"]["index"])


# Data Loader
from torch.utils.data import DataLoader
test_loader = DataLoader(ds_test, batch_size = params["batch_size"])

# Model
feat_count = len(ds["metadata"]["features"])
col_count = feat_count + len(ds["metadata"]["input_only_features"])

model = Decoder(
    hidden_dim=params["hidden_dim"],
    seq_len=SEQ_LEN,
    corpus_size=feat_count,
    n_heads=params["n_attn_heads"],
    n_blocks=params["n_blocks"],
    dropout=params["dropout"],
    use_embedding=False,
    embedding_replacement=torch.nn.Linear(col_count, params["hidden_dim"])
).to(DEVICE)

try:
    model.load_state_dict(torch.load(f"./model/{checkpoint}.pth", map_location="cpu", weights_only=True))
except FileNotFoundError:
    raise FileNotFoundError(f"Checkpoint at {checkpoint} epochs not found")
model.to(DEVICE)

criterion = torch.nn.SmoothL1Loss()

print("Device:", next(model.parameters()).device)
print("Parameter Count:", f"{sum(p.numel() for p in model.parameters()):,}")
print("Allocated Memory:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("Reserved memory:", torch.cuda.memory_reserved() / 1024**3, "GB")



def load_tensor(entities, seq, seq_len=SEQ_LEN):
    regional_data = [] 

    for entity, sid in zip(entities, seq):
        sid = int(sid)
        entity = entity.item() if hasattr(entity, "item") else entity

        regional_data.append(torch.tensor(df.loc[entity].loc[sid:sid + seq_len].to_numpy()))


    regional_data = torch.stack(regional_data).to(DEVICE, torch.float32)
    return regional_data


# Eval Loop
model.eval()
test_loss = 0
test_se = torch.zeros([feat_count, len(SEQ_PRED)]).to(DEVICE)
seq_to_idx = {sid: i for i, sid in enumerate(ds["metadata"]["seq_starts"])}
test_count = 0
with torch.no_grad():
    for entities, seq_id in tqdm(test_loader, desc=f"Test"):
        regional_data = load_tensor(entities, seq_id)
        pred = model(regional_data[:, :-1])
        ground_truth = regional_data[:, -1, 4:]
        pred = pred[:, -1, :]
    

        loss = criterion(pred, ground_truth)
        test_loss += loss.item()

        # Track each feature and seq id squared errors
        sq_errs = ((pred - ground_truth)**2)
        for se, sid in zip(sq_errs, seq_id):
            test_se[:, seq_to_idx[sid.item()]] += se

        test_count += pred.shape[0]

test_se = test_se.cpu()

city_occurence_count = test_count/len(ds["metadata"]["test_city_id"])
city_count = len(ds["metadata"]["test_city_id"])

global_rmse = (test_se.sum().item()/(city_occurence_count*city_count*feat_count))**0.5
features_rmse = (test_se.sum(dim=1).numpy()/(city_occurence_count*city_count))**0.5
seq_rmse = (test_se.sum(dim=0).numpy()/(city_occurence_count*feat_count))**0.5
test_rmse = (test_se.numpy()/city_occurence_count)**0.5

test_loss = test_loss/len(test_loader)

print(f"Test Loss: {test_loss}, RMSE = {global_rmse}")




# ----- Visualise prediction result for a sample city -----

# Pick data
used_data = ds_test[0]

pred_test = sorted(
    [x for x in ds_test if x[0] == used_data[0]],
    key=lambda x: x[1]
)

result = []

# Inference
with torch.no_grad():
    for entities, years in tqdm(pred_test, desc=f"Inference"):
        regional_data = load_tensor([entities], [years])
        pred = model(regional_data[:, :-1])
        ground_truth = regional_data[:, -1, 4:]
        pred = pred[:, -1]
        result.append(pred)

result = torch.stack(result).squeeze(1)

# Load  Scaler
from joblib import load

scaler = load("./model/scaler.pkl")

# Inverse Transform Prediction
data = result.detach().cpu().numpy()

filler = np.zeros((data.shape[0], 4))
data = np.concatenate([filler, data], axis=1)

data = scaler.inverse_transform(data)
data = data[:, 4:]

all_seq_len = len(SEQ_PRED) + SEQ_LEN
selected_seq = pred_test[0]

# Inverse Transform Ground Truth
ground_truth = load_tensor([selected_seq[0]], [selected_seq[1]], seq_len=all_seq_len)
ground_truth = ground_truth.squeeze(0).detach().cpu().numpy()
ground_truth = scaler.inverse_transform(ground_truth)
ground_truth = ground_truth[:, 4:]


# Visulaisation
import matplotlib.pyplot as plt

data_range = np.arange(SEQ_LEN, all_seq_len, 1)
gt_range = np.arange(0, all_seq_len, 1)

fig, ax = plt.subplots(4, 5, figsize=(25, 8))

for i in range(19):
    ax[i//5, i%5].plot(data_range, data[:, i], label = "pred")
    ax[i//5, i%5].plot(gt_range, ground_truth[:, i], label = "gt")
    ax[i//5, i%5].set_title(ds["metadata"]["features"][i])
    ax[i//5, i%5].set_xlim(0, all_seq_len+11)

handles, labels = ax[0, 0].get_legend_handles_labels()

# Create a figure-level legend
fig.suptitle(f"Prediction for {selected_seq[0].replace(' | ', ', ')}", fontsize=20)
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=20, bbox_to_anchor=(0.5, 0.94))

plt.tight_layout(rect=[0, 0, 1, 0.88])  # Leave space for the legend
plt.savefig(f"./results/{selected_seq[0].replace(' | ', '-').replace(' ', '_')}")


# ----- RMSE Visualisation -----

fig, ax = plt.subplots(4, 5, figsize=(25, 8))

for i in range(19):
    ax[i//5, i%5].plot(SEQ_PRED, test_rmse[i, :], label = "pred")
    ax[i//5, i%5].hlines(features_rmse[i], min(SEQ_PRED), max(SEQ_PRED), color="red", label = "average")
    ax[i//5, i%5].set_title(ds["metadata"]["features"][i])
    ax[i//5, i%5].set_xlim(min(SEQ_PRED), max(SEQ_PRED)+11)

ax[3, 4].plot(SEQ_PRED, seq_rmse, label = "pred")
ax[3, 4].hlines(global_rmse, min(SEQ_PRED), max(SEQ_PRED), color="red", label = "average")
ax[3, 4].set_title("All Features Combined")
ax[3, 4].set_xlim(min(SEQ_PRED), max(SEQ_PRED)+11)

handles, labels = ax[0, 0].get_legend_handles_labels()

# Create a figure-level legend
fig.suptitle(f"Normalized RMSE for each features", fontsize=20)
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=20, bbox_to_anchor=(0.5, 0.94))

plt.tight_layout(rect=[0, 0, 1, 0.88])  # Leave space for the legend
plt.savefig(f"./results/rmse")


# ----- Log Evaluation -----
# Read log file
metrics = pd.read_csv("./results/training_log.csv")
n = np.arange(1, metrics["epoch"].max()+2)

# Create Plot
fig, ax1 = plt.subplots()

ax1.plot(n, metrics["train_loss"], "C0", label="Train Loss")
ax1.plot(n, metrics["train_rmse"], "C1", label="Train RMSE")

ax1.plot(n, metrics["val_loss"], "C0:", label="Val Loss")
ax1.plot(n, metrics["val_rmse"], "C1:", label="Val RMSE")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Value")
ax1.set_yscale("log")

ax2 = ax1.twinx()

ax2.plot(n, metrics["lr"], "C2-", label="Learning Rate")
ax2.set_ylabel("Learning Rate")
ax2.set_yscale("log")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
plt.title("Training Metrics")

plt.savefig(f"./results/training_metrics")