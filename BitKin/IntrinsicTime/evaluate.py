

import torch
import pickle
import numpy as np

from IntrinsicTime.dc_model import DcModel, DcDataset
from IntrinsicTime.plot import Plot
from IntrinsicTime.vis_prediction import VisPrediction

if __name__ == '__main__':
    with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
        data = pickle.load(f)

    deltas, order_books, runners, runner_clock, targets, features = data

    features = torch.from_numpy(features).float()

    print('deltas', deltas)

    device = 'cpu'
    dc_model = DcModel(n_timesteps=features.shape[2], device=device)
    dc_model.load_state_dict(torch.load('model'))
    dc_model.eval()

    features = features[:, 0:19, :]
    #print(features[:, 0:19, :].shape)

    #quit()
    predictions = dc_model(features).detach().numpy()
    #print(predictions[200])
    #quit()
    predictions = ((1 - predictions) * 255).astype('int').clip(0, 255)

    #print(predictions[200])
    #quit()

    target_direction = targets
    measured_direction = features[:, 0:len(deltas), :].numpy()
    TMV = features[:, 0:len(deltas), :].numpy()
    RET = features[:, 0:len(deltas), :].numpy()
    #TMV = features[:, len(deltas):2 * len(deltas), :].numpy()
    #RET = features[:, 2 * len(deltas):3 * len(deltas), :].numpy()

    #print("target_direction", target_direction.shape)

    #target_direction(19, 9449)
    #measured_direction(19, 9449)
    #print("target_direction", target_direction.shape)
    #print("measured_direction", measured_direction.shape)
    #quit()

    vis = VisPrediction()
    vis.start()

    x = np.arange(len(runner_clock.ie_times))
    plot = Plot()
    plot.plot((x, runner_clock.ie_prices, target_direction, measured_direction))
    plot.show()

    while True:
        cmd, payload = plot.get()
        if cmd == 'quit':
            break

        elif cmd == 'x' and payload is not None:
            feature_length = measured_direction.shape[2]

            x = max(min(int(payload) - feature_length, measured_direction.shape[0]), 0)

            dc_feature = measured_direction[x]
            target_feature = target_direction[x]
            prediction = predictions[x]

            vis.update_data(dc_feature, target_feature, prediction)


            """
            measured_direction (11739, 19, 10)
            ---
            dc_feature(19, 100)
            tmv_feature(19, 100)
            ret_feature(19, 100)
            """

            """
            feature_length = 100
            x = int(payload) - feature_length
            max_x = measured_direction.shape[1] - feature_length
            x = max(min(x, max_x), 0)
            md_feature = measured_direction[:, x:x + feature_length]
            td_feature = target_direction[:, x:x + feature_length]
            tmv_feature = TMV[:, x:x + feature_length]
            ret_feature = RET[:, x:x + feature_length]

            dc_feature = measured_direction[:, x:x + feature_length].copy()
            #dirs = direction_changes[:, x:x + feature_length]
            #for runner_idx in range(dc_feature.shape[0]):
            #    idx = feature_length - 1
            #    while idx > -1 and dirs[runner_idx, idx] == 0:
            #        idx -= 1
            #    if idx >= 0:
            #        dc_feature[runner_idx, 0: idx] = td_feature[runner_idx, 0: idx]

            vis.update_data(dc_feature, tmv_feature, ret_feature)
            """

    plot.shutdown()
