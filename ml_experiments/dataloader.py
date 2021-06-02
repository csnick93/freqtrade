import numpy as np
import os
import pathlib
import torch

TARGET_DIR = pathlib.Path(
    '/Users/nicolasvonroden/Data/freqtrade/prediction_data')


def get_dataloaders(batch_size: int, cnn: bool):
    data_files = os.listdir(TARGET_DIR)
    num_train = int(len(data_files) * 0.7)
    num_val = int(len(data_files) * 0.1)

    train_files = data_files[:num_train]
    val_files = data_files[num_train:num_train + num_val]
    test_files = data_files[num_train + num_val:]

    if cnn:
        X_train = np.expand_dims(np.concatenate(
            [np.load(TARGET_DIR / f)['X'] for f in train_files]),
                                 axis=1)
        X_val = np.expand_dims(np.concatenate(
            [np.load(TARGET_DIR / f)['X'] for f in val_files]),
                               axis=1)
        X_test = np.expand_dims(np.concatenate(
            [np.load(TARGET_DIR / f)['X'] for f in test_files]),
                                axis=1)
    else:
        X_train = np.concatenate(
            [np.load(TARGET_DIR / f)['X'] for f in train_files])
        X_val = np.concatenate(
            [np.load(TARGET_DIR / f)['X'] for f in val_files])
        X_test = np.concatenate(
            [np.load(TARGET_DIR / f)['X'] for f in test_files])

    y_train = np.concatenate(
        [np.load(TARGET_DIR / f)['y'] for f in train_files])
    y_val = np.concatenate([np.load(TARGET_DIR / f)['y'] for f in val_files])
    y_test = np.concatenate([np.load(TARGET_DIR / f)['y'] for f in test_files])

    dataset_train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).type(torch.long))
    dataset_val = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).type(torch.long))
    dataset_test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).type(torch.long))

    dataloaders = {
        'train':
        torch.utils.data.DataLoader(dataset_train,
                                    batch_size=batch_size,
                                    num_workers=4),
        'val':
        torch.utils.data.DataLoader(dataset_val,
                                    batch_size=batch_size,
                                    num_workers=4),
        'test':
        torch.utils.data.DataLoader(dataset_test,
                                    batch_size=batch_size,
                                    num_workers=4)
    }
    dataset_sizes = {
        'train': X_train.shape[0],
        'val': X_val.shape[0],
        'test': X_test.shape[0]
    }

    return dataloaders, dataset_sizes
