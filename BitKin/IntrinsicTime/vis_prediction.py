
import numpy as np
import cmasher
from multiprocessing import Process, Pipe


def vis_process(conn):
    import pygame

    target_direction = np.random.randint(2, size=(19, 200))
    direction_change = np.random.randint(2, size=(19, 200))
    predictions = np.random.randint(2, size=(19, 200))
    #tmv = np.random.randint(2, size=(19, 200))
    #ret = np.random.randint(2, size=(19, 200))

    pygame.init()

    clock = pygame.time.Clock()
    display_size = (600, 900)
    game_display = pygame.display.set_mode(display_size)
    pygame.display.set_caption('Directions')
    display_rect = pygame.Rect((0, 0), display_size)
    pygame.display.update()

    colormap = cmasher.take_cmap_colors('cmr.watermelon', 256, return_fmt='int')

    running = True
    while running:
        while conn.poll():
            cmd, payload = conn.recv()
            if cmd == 'quit':
                break
            elif cmd == 'update_data':
                direction_change, target_direction, predictions = payload

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == ord('q'):
                    running = False

        game_display.fill(color=(255, 255, 255))

        def draw_directions(directions, area):
            shape = directions.shape
            rect_size = ((area[2] - area[0]) / shape[1], (area[3] - area[1]) / shape[0])
            for ypos in range(shape[0]):
                y = area[1] + ypos * rect_size[1]
                for xpos in range(shape[1]):
                    x = area[0] + xpos * rect_size[0]
                    if directions[shape[0] - 1 - ypos, xpos] == 0:
                        color = (250, 250, 250)
                    else:
                        color = (40, 90, 40)
                    game_display.fill(color, (x, y, rect_size[0] + 1, rect_size[1] + 1))

        def draw_predictions(values, area):
            rect_size = ((area[2] - area[0]) / values.shape[1], (area[3] - area[1]) / values.shape[0])
            for ypos in range(values.shape[0]):
                y = area[1] + ypos * rect_size[1]
                for xpos in range(values.shape[1]):
                    x = area[0] + xpos * rect_size[0]

                    value = values[values.shape[0] - 1 - ypos, xpos]
                    #value = int(-value * 128 + 128)
                    #value = max(min(value, 255), 0)

                    color = colormap[value]

                    # if directions[shape[0] - 1 - ypos, xpos] == 0:
                    #    color = (250, 250, 250)
                    # else:
                    #    color = (40, 90, 40)
                    game_display.fill(color, (x, y, rect_size[0] + 1, rect_size[1] + 1))

        draw_directions(direction_change, (10, 10, display_size[0] - 10, 300 - 10))
        draw_directions(target_direction, (10, 310, display_size[0] - 10, 600 - 10))
        draw_predictions(predictions, (10, 610, display_size[0] - 10, 900 - 10))

        pygame.display.update()

        clock.tick(10)

    pygame.quit()


class VisPrediction:
    def __init__(self):
        self.conn, conn_remote = Pipe()
        self.p = Process(target=vis_process, args=(conn_remote, ))

    def update_data(self, direction_change, target_direction, predictions):
        self.conn.send(('update_data', (direction_change, target_direction, predictions)))

    def start(self):
        self.p.start()


if __name__ == '__main__':
    measured_direction = np.random.randint(2, size=(19, 200))
    target_direction = np.random.randint(2, size=(19, 200))

    vis = VisPrediction()
    vis.start()
    vis.update_data(direction_change=measured_direction, target_direction=target_direction, predictions=predictions)
