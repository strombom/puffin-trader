import os
import pickle
import subprocess
import numpy as np
import pandas as pd
from fastai.tabular.all import *


def train():
    training_path = 'C:/BitBotLiveV1/training_data/'
    models_path = 'C:/BitBotLiveV1/models/'
    # training_path = 'C:/BitBot/training_data/'
    # models_path = 'C:/BitBot/models/'
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    train_on_all_data = True

    print("Load data")

    #timestamps_train, timestamps_valid = set(), set()
    symbols = set()

    dfs = []
    #if not random_split:
    splits = [[], []]

    start_idx = 0
    for filename in os.listdir(training_path):
        symbols.add(filename.split('_')[0])
        df = pd.read_csv(training_path + filename)
        dfs.append(df)

        #if not random_split:
        split_idx = start_idx + int(df.shape[0] * 0.95)
        end_idx = start_idx + df.shape[0]
        if train_on_all_data:
            splits[0].extend(list(range(start_idx, end_idx)))
        else:
            splits[0].extend(list(range(start_idx, split_idx)))
        splits[1].extend(list(range(split_idx, end_idx)))
        start_idx += df.shape[0]

    df = pd.concat(dfs)

    """    
    dfs_train, dfs_valid = [], []
    for filename in os.listdir(training_path):
        if timestamp_train in filename and 'train' in filename:
            df = pd.read_csv(training_path + filename)
            dfs_train.append(df)
        elif timestamp_valid in filename and 'valid' in filename:
            df = pd.read_csv(training_path + filename)
            dfs_valid.append(df)
    dfs_train, dfs_valid = pd.concat(dfs_train), pd.concat(dfs_valid)
    return dfs_train, dfs_valid
    """

    """
    def make_splits(dfs_train, dfs_valid):
        len_train, len_valid = dfs_train.shape[0], dfs_valid.shape[0]
        splits = [
            list(range(0, len_train)),
            list(range(len_train, len_train + len_valid))
        ]
        df = pd.concat([dfs_train, dfs_valid])
        return df, splits
    """

    #if random_split:
    #    splits = RandomSplitter(valid_pct=0.2)(range_of(df))

    print("Make column names")
    y_count = 8
    cat_names = list(df.columns)[-len(symbols)-y_count:-y_count]
    cont_names = list(df.columns)[2:-len(symbols)-y_count]
    y_names = list(df.columns)[-y_count:]

    print("Make tabular dataset")
    to = TabularPandas(df, procs=[Categorify], cat_names=cat_names, cont_names=cont_names, y_names=y_names, splits=splits)

    print("Make dataloader")
    dataloader = to.dataloaders(bs=2**10)

    print("Fit one cycle")
    learn = tabular_learner(dataloader, layers=[500, 400, 300, 200, 150, 100, 50], metrics=rmse)
    learn.fit_one_cycle(5, lr_max=1e-4)

    """
    print("Make predictions")
    dl_train = DataLoader(dataset=df.iloc[splits[0]])
    df_val = DataLoader(dataset=df.iloc[splits[1]])
    df_train, df_val = df.iloc[splits[0]], df.iloc[splits[1]]
    dl_train = learn.dls.test_dl(df_train)
    dl_val = learn.dls.test_dl(df_val)
    pred_train, gt_train = learn.get_preds(dl=dl_train)
    pred_val, gt_val = learn.get_preds(dl=dl_val)
    os.makedirs('E:/BitBot/preds/', exist_ok=True)
    with open(f'E:/BitBot/preds/preds.pickle', 'wb') as f:
        pickle.dump({
            'pred_train': pred_train.squeeze(),
            'gt_train': gt_train.squeeze(),
            'pred_val': pred_val.squeeze(),
            'gt_val': gt_val.squeeze()
        }, f)
    """

    print("Save model")
    learn.export(models_path + f"/model.pickle")


if __name__ == '__main__':
    #train()
    #quit()

    while True:
        print("Remove old training data")
        with os.scandir("C:/BitBotLiveV1/training_data") as entries:
            for entry in entries:
                if not entry.is_dir() and not entry.is_symlink():
                    os.remove(entry.path)

        os.chdir("C:/development/github/puffin-trader/BitBot_Live_v1/x64/Release/")
        make_data_executable = "BitBot_Live_v1.exe"
        print("Run", make_data_executable)
        result = subprocess.run(args=[make_data_executable], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)

        if result.returncode == 0:
            train()
            time.sleep(60 * 60)
        else:
            time.sleep(1 * 60)
