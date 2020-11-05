
import torch
import pickle

from dc_model import DcModel, DcDataset


with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
    data = pickle.load(f)

deltas, order_books, runners, runner_clock, targets, features = data

features = torch.from_numpy(features).float()

print(deltas)

device = 'cpu'
dc_model = DcModel(n_timesteps=features.shape[2], device=device)
dc_model.load_state_dict(torch.load('model'))
dc_model.eval()

#xshape torch.Size([128, 58, 10])
print(features[100].unsqueeze(0).shape)
#quit()
y_hat = dc_model(features[303].unsqueeze(0))
print(y_hat.shape)
print(y_hat)
