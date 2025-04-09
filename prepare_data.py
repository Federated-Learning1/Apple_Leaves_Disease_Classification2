# prepare_data.py

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
NUM_CLIENTS = 3
DATA_DIR = "plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
CLIENTS_BASE_DIR = "clients"
VAL_DIR = "val"

# Make class mapping
def get_label(row):
    labels = ["healthy", "multiple_diseases", "rust", "scab"]
    return row[labels].idxmax()

def split_data(df, num_clients=NUM_CLIENTS, val_split=0.2):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["label"] = df.apply(get_label, axis=1)

    train_df, val_df = train_test_split(df, test_size=val_split, stratify=df["label"], random_state=42)

    clients_data = [train_df.iloc[i::num_clients].reset_index(drop=True) for i in range(num_clients)]
    return clients_data, val_df

def save_client_data(client_id, df):
    client_dir = os.path.join(CLIENTS_BASE_DIR, f"client{client_id}")
    images_out = os.path.join(client_dir, "images")
    os.makedirs(images_out, exist_ok=True)

    df["image"] = df["image_id"] + ".jpg"
    df[["image", "label"]].to_csv(os.path.join(client_dir, "data.csv"), index=False)

    for fname in df["image"]:
        src = os.path.join(IMAGES_DIR, fname)
        dst = os.path.join(images_out, fname)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)

def save_val_data(val_df):
    os.makedirs(os.path.join(VAL_DIR, "images"), exist_ok=True)
    val_df["image"] = val_df["image_id"] + ".jpg"
    val_df[["image", "label"]].to_csv(os.path.join(VAL_DIR, "val.csv"), index=False)

    for fname in val_df["image"]:
        src = os.path.join(IMAGES_DIR, fname)
        dst = os.path.join(VAL_DIR, "images", fname)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)

def main():
    df = pd.read_csv(TRAIN_CSV)
    clients_data, val_df = split_data(df)

    # Save each client data
    for i, client_df in enumerate(clients_data):
        save_client_data(i + 1, client_df)

    # Save validation set
    save_val_data(val_df)

if __name__ == "__main__":
    main()
