

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


norm = torchvision.transforms.Normalize(mean=[ 0.5533,  0.5563,  0.5872,  0.5788,  0.5771,  0.5916,  0.5669,  0.5585,
                                               0.5965,  0.5861,  0.6195,  0.6159,  0.5876,  0.6443,  0.5022,  0.5607,
                                               0.5544,  0.7234,  0.6282,  0.2103,  0.2046,  0.2499,  0.2619,  0.2328,
                                               0.2498,  0.2324,  0.2011,  0.2571,  0.2555,  0.2775,  0.2529,  0.2004,
                                               0.3126,  0.1285,  0.1803,  0.1553,  0.3783,  0.2942,  0.0928,  0.0841,
                                               0.0964,  0.1013,  0.0788,  0.0556,  0.0303,  0.0141,  0.0228,  0.0410,
                                               0.0423,  0.0336,  0.0048,  0.0041, -0.0105, -0.0060, -0.0052,  0.0145,
                                               0.0092,  0.5000],
                                        std=[  0.4972, 0.4968, 0.4924, 0.4938, 0.4940, 0.4916, 0.4955, 0.4966, 0.4906,
                                               0.4926, 0.4855, 0.4864, 0.4923, 0.4788, 0.5000, 0.4963, 0.4971, 0.4473,
                                               0.4833, 0.9618, 0.9478, 0.9080, 0.8957, 0.8811, 0.8559, 0.8667, 0.8335,
                                               0.8469, 0.8096, 0.8405, 0.7865, 0.7515, 0.7853, 0.8256, 0.7923, 0.7239,
                                               0.6611, 0.6207, 0.7900, 0.7303, 0.6827, 0.6167, 0.5790, 0.5346, 0.4577,
                                               0.4234, 0.3888, 0.3376, 0.3059, 0.2776, 0.2552, 0.1699, 0.1398, 0.1294,
                                               0.1117, 0.0615, 0.0539, 0.3192])

norm = torchvision.transforms.Normalize(mean=[ 0.5533,  0.5563,  0.5872,  0.5788,  0.5771,  0.5916,  0.5669,  0.5585,
                                               0.5965,  0.5861,  0.6195,  0.6159,  0.5876,  0.6443,  0.5022,  0.5607,
                                               0.5544,  0.7234,  0.6282],
                                        std=[  0.4972, 0.4968, 0.4924, 0.4938, 0.4940, 0.4916, 0.4955, 0.4966, 0.4906,
                                               0.4926, 0.4855, 0.4864, 0.4923, 0.4788, 0.5000, 0.4963, 0.4971, 0.4473,
                                               0.4833])


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * ( torch.tanh(F.softplus(x)))


class DcModelInner(nn.Module):
    def __init__(self):
        super(DcModelInner, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(19, 200),  # 77
            torch.nn.AlphaDropout(0.3),
            Mish(),
            #torch.nn.Linear(500, 400),
            #torch.nn.AlphaDropout(0.3),
            #torch.nn.ReLU6(),
            #torch.nn.Linear(400, 300),
            #torch.nn.AlphaDropout(0.3),
            #torch.nn.ReLU6(),
            #torch.nn.Linear(300, 200),
            #torch.nn.AlphaDropout(0.3),
            #torch.nn.ReLU6(),
            #torch.nn.Linear(200, 100),
            #torch.nn.AlphaDropout(0.3),
            #torch.nn.ReLU6(),
            torch.nn.Linear(200, 19),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        #x = torch.cat((x, prev_prediction), dim=1)
        x = self.model.forward(x)
        return x


class DcModel(nn.Module):
    def __init__(self, n_timesteps, device):
        super(DcModel, self).__init__()
        self.n_timesteps = n_timesteps
        self.inner = DcModelInner()
        self.device = device
        self.to(device)

    def forward(self, x):

        #print("x", x.shape)
        #x = norm.forward(torch.reshape(x, shape=(x.shape[0], x.shape[1], x.shape[2], 1))).squeeze(3)

        prediction = torch.empty((x.shape[0], 19, self.n_timesteps)).to(self.device)
        #prediction[:, :, 0] = x[:, 0:19, 0]
        #prediction.names = ['C', 'T']

        #prediction[:, :, 0] = self.inner.forward(x[:, :, 0], x[:, 0:19, 0])
        for step_idx in range(self.n_timesteps):
            prediction[:, :, step_idx] = self.inner.forward(x[:, :, step_idx])
            # self.inner.forward(x[:, :, step_idx], x[:, :, step_idx])

        return prediction

        """
        #print("x", x.shape)
        x = norm.forward(torch.reshape(x, shape=(x.shape[0], x.shape[1], x.shape[2], 1))).squeeze(3)

        prediction = torch.empty((x.shape[0], 19, self.n_timesteps)).to(self.device)
        prediction[:, :, 0] = x[:, 0:19, 0]
        #prediction.names = ['C', 'T']

        prediction[:, :, 0] = self.inner.forward(x[:, :, 0], x[:, 0:19, 0])
        for step_idx in range(1, self.n_timesteps):
            #print("1", x[:, :, step_idx])
            #print("2", prediction[:, :, step_idx - 1])
            #quit()
            prediction[:, :, step_idx] = self.inner.forward(x[:, :, step_idx], prediction[:, :, step_idx - 1])

        return prediction
        """


class DcDataset(Dataset):
    def __init__(self, features_np, targets_np, device):
        self.features = torch.from_numpy(features_np).float().to(device)
        self.targets = torch.from_numpy(targets_np).float().to(device)
        #self.targets.names = ['N', 'C', 'T']
        #self.features.names = ['N', 'C', 'T']

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)



