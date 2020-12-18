
import pickle
import numpy

from IntrinsicTime.runner import Direction
from Plot.plot_tmvr import Plot
from Plot.vis_features import VisFeatures


if __name__ == '__main__':
    with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
        deltas, order_books, runners, runner_clock, clock_TMV, clock_R = pickle.load(f)

    print(f'Deltas({len(deltas)}): {deltas}')

    print(f"TMV {numpy.min(clock_TMV)} {numpy.max(clock_TMV)}")
    print(f"R {numpy.min(clock_R)} {numpy.max(clock_R)}")

    measured_direction = numpy.zeros((len(deltas), len(runner_clock.ie_times)))
    direction_changes = numpy.zeros((len(deltas), len(runner_clock.ie_times)))

    # Measured direction
    for idx_runner, runner in enumerate(runners):
        direction = Direction.up
        idx_dc = 0
        for idx_clock, timestamp in enumerate(runner_clock.ie_times):
            while idx_dc < len(runner.dc_times) and runner.dc_times[idx_dc] < timestamp:
                idx_dc += 1
                if direction == Direction.up:
                    direction = Direction.down
                else:
                    direction = Direction.up
            if idx_dc >= len(runner.dc_times):
                break
            if direction == Direction.up:
                measured_direction[idx_runner, idx_clock] = 1
            else:
                measured_direction[idx_runner, idx_clock] = 0

    # Direction changes
    for idx_runner, runner in enumerate(runners):
        idx_dc = 0
        for idx_clock, timestamp in enumerate(runner_clock.ie_times):
            found = False
            while idx_dc < len(runner.dc_times) and runner.dc_times[idx_dc] < timestamp:
                idx_dc += 1
                found = True

            if idx_dc >= len(runner.dc_times):
                break

            if found: #runner.dc_times[idx_dc] == timestamp:
                direction_changes[idx_runner, idx_clock] = 1

    x = numpy.arange(len(runner_clock.ie_times))

    vis = VisFeatures()
    vis.start()
    plot = Plot()
    plot.plot((x, runner_clock.ie_prices))
    plot.show()
    while True:
        cmd, payload = plot.get()
        if cmd == 'quit':
            break
        elif cmd == 'x' and payload is not None:
            feature_length = 100
            x = int(payload) - feature_length
            max_x = measured_direction.shape[1] - feature_length
            x = max(min(x, max_x), 0)
            md_feature = measured_direction[:, x:x + feature_length]
            #td_feature = target_direction[:, x:x + feature_length]
            tmv_feature = clock_TMV[:, x:x + feature_length]
            ret_feature = clock_R[:, x:x + feature_length]

            dc_feature = measured_direction[:, x:x + feature_length].copy()
            dirs = direction_changes[:, x:x + feature_length]
            for runner_idx in range(dc_feature.shape[0]):
                idx = feature_length - 1
                while idx > -1 and dirs[runner_idx, idx] == 0:
                    idx -= 1
                if idx >= 0:
                    dc_feature[runner_idx, 0: idx] = md_feature[runner_idx, 0: idx]

            #print("dc_feature", dc_feature.shape)
            #print("tmv_feature", tmv_feature.shape)
            #print("ret_feature", ret_feature.shape)
            vis.update_data(dc_feature, tmv_feature, ret_feature)
    plot.shutdown()


