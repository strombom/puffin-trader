
import numpy as np
import sys
import csv

episode_idx = sys.argv[1]
titles = []
rows = []

with open('C:\\development\\github\\puffin-trader\\tmp\\log\\cartpole_' + episode_idx + '.csv') as csvfile:
    reader = csv.reader(csvfile)
    is_first = True
    for row in reader:
        if is_first:
            titles = row
            is_first = False
        else:
            rows.append([float(i) for i in row])

rows = np.array(rows)

from math import sin, cos
import gizeh as gz
import moviepy.editor as mpy


width, height = 500, 300
fps = 20
duration = rows.shape[0] / fps
first = True

def make_frame(time):
    row_idx = int(time * fps)
    print("time", row_idx, time)

    cart_position, cart_velocity, pole_angle, pole_velocity = rows[row_idx]
    row_idx += 1

    surface = gz.Surface(width, height, bg_color=(1,1,1))

    x, y = width * (0.5 + cart_position * 5), height * 0.8

    rect = gz.rectangle(lx=.15*width, ly=.15*height, xy=(x, y), fill=(0,1,0))
    rect.draw(surface)

    pole_length = height * 0.5
    dx = sin(pole_angle) * pole_length
    dy = -cos(pole_angle) * pole_length

    line = gz.polyline(points=[(x,y), (x + dx,y + dy)], stroke_width=10,
                         stroke=(1,0,0), fill=(1,0,0))
    line.draw(surface)

    return surface.get_npimage()

clip = mpy.VideoClip(make_frame, duration=duration)
clip.write_videofile('cartpole/cartpole_' + episode_idx + '.mp4', fps=25) # Many options...

