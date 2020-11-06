import pickle

import statistics
import torch
#import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from dc_model import DcModel, DcDataset

#import torch.nn.functional as F
#import numpy as np
#from torch.utils.tensorboard import SummaryWriter
#from torch.autograd import Variable

import warnings
warnings.simplefilter("ignore", UserWarning)
#warnings.simplefilter("ignore", torch.TracerWarning)


with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
    data = pickle.load(f)

deltas, order_books, runners, runner_clock, targets, features = data


def make_train_step(model, loss_fn, optimizer):
    def train_step_fn(x, y):
        model.train()
        y_hat = model(x)
        loss = loss_fn(y, y_hat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step_fn


# (batches, feature_types, deltas, timesteps)
#deltas, order_books, runners, runner_clock, target_direction, measured_direction, TMV, RET

#n_batches = 1
#n_deltas = len(deltas)
#n_feature_types = 3
#n_timesteps =

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(f'device: {device}')

learning_rate = 1e-2
n_epochs = 50

dc_model = DcModel(n_timesteps=features.shape[2], device=device)

n_train = int(features.shape[0] * 0.8)
n_validation = features.shape[0] - n_train
dataset_train = DcDataset(features[0:n_train], targets[0:n_train], device)
dataset_validation = DcDataset(features[n_train:], targets[n_train:], device)

dataloader_train = DataLoader(dataset=dataset_train, batch_size=128, shuffle=True)
dataloader_validation = DataLoader(dataset=dataset_validation, batch_size=1024, shuffle=False)

loss_fn = torch.nn.MSELoss(reduction='mean')
#loss_fn = torch.nn.MultiLabelMarginLoss(reduction='mean')
optimizer = optim.SGD(dc_model.parameters(), lr=learning_rate)
train_step = make_train_step(model=dc_model, loss_fn=loss_fn, optimizer=optimizer)

for epoch in range(n_epochs):
    losses = []
    for x, y in dataloader_train:
        #print("xshape", x.shape)
        #quit()
        loss = train_step(x, y)
        losses.append(loss)

    with torch.no_grad():
        val_losses = []
        for x, y in dataloader_validation:
            dc_model.eval()
            y_hat = dc_model(x)
            val_loss = loss_fn(y, y_hat).item()
            val_losses.append(val_loss)

    print(f'{epoch} train({statistics.mean(losses)}) val({statistics.mean(val_losses)})')


torch.save(dc_model.state_dict(), "model")


print("deltas", len(deltas), deltas)
print("targets", targets.shape)
print("features", features.shape)



quit()


# Normalization
if False:
    features_mean = torch.mean(features, dim=(0, 2))
    features_std = torch.std(features, dim=(0, 2))
    print("features_mean", features_mean.shape, features_mean)
    print("features_std", features_std.shape, features_std)

print("feature", dataset[302][0])
print("prediction", dataset[302][1])
prediction = net.forward(dataset[302][0])
print("target", prediction)

quit()

#loss_fn = torch.nn.MSELoss(reduction='sum')

#loss = loss_fn(prediction, targets[100])
#net.zero_grad()
#loss.backward()



#writer = SummaryWriter('runs/experiment_1')
#writer.add_graph(net, features[100])
#writer.close()

