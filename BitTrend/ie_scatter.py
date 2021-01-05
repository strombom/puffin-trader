
from IntrinsicTime.runner import Direction


def make_ie_scatters(runner):
    ie_scatters = []
    colors = ['xkcd:green', 'xkcd:light red', 'xkcd:lime', 'xkcd:baby pink']
    for i in range(4):
        marker = '+'  # '^' if i % 2 == 0 else 'v'
        size = 40
        color = colors[i]
        ie_scatters.append({'marker': marker, 'size': size, 'color': color, 'x': [], 'y': []})

    ie_overshoots = {'x': [], 'y': []}
    idx_dc, idx_os = 0, 0
    direction = Direction.down
    for idx_ie, timestamp in enumerate(runner.ie_times):
        turn = False
        while idx_os + 1 < len(runner.os_times) and timestamp > runner.os_times[idx_os + 1]:
            idx_os += 1
            ie_overshoots['x'].append(idx_ie - 1)
            ie_overshoots['y'].append(runner.os_prices[idx_os])
            turn = True
            if runner.ie_prices[idx_ie] > runner.os_prices[idx_os]:
                direction = Direction.up
            else:
                direction = Direction.down

        while idx_dc + 1 < len(runner.dc_times) and timestamp >= runner.dc_times[idx_dc + 1]:
            idx_dc += 1
        if idx_dc >= len(runner.dc_times):
            break

        if direction == Direction.up:
            scatter_idx = 0
        else:
            scatter_idx = 1

        if idx_ie + 1 < len(runner.ie_times):
            if runner.ie_times[idx_ie + 1] == runner.ie_times[idx_ie]:
                scatter_idx += 2  # Free fall

        ie_scatters[scatter_idx]['x'].append(idx_ie)
        ie_scatters[scatter_idx]['y'].append(runner.ie_prices[idx_ie])

    return ie_scatters, ie_overshoots
