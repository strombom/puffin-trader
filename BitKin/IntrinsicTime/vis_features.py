
import pygame
import numpy as np
from multiprocessing import Process, Pipe


def vis_process(conn):
    measured_direction = np.random.randint(2, size=(19, 200))
    target_direction = np.random.randint(2, size=(19, 200))

    pygame.init()

    clock = pygame.time.Clock()
    display_size = (600, 400)
    game_display = pygame.display.set_mode(display_size)
    pygame.display.set_caption('Directions')
    display_rect = pygame.Rect((0, 0), display_size)
    pygame.display.update()

    running = True
    while running:
        if conn.poll():
            cmd, payload = conn.recv()
            if cmd == 'quit':
                break

            elif cmd == 'update_data':
                measured_direction, target_direction = payload

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
                    if directions[ypos, xpos] == 0:
                        color = (240, 10, 10)
                    else:
                        color = (10, 230, 10)
                    game_display.fill(color, (x, y, rect_size[0] + 1, rect_size[1] + 1))

        draw_directions(measured_direction, (10, 10, display_size[0] - 10, 200 - 10))
        draw_directions(target_direction, (10, 210, display_size[0] - 10, 400 - 10))

        pygame.display.update()

        clock.tick(20)

    pygame.quit()


class VisFeatures:
    def __init__(self):
        self.conn, conn_remote = Pipe()
        self.p = Process(target=vis_process, args=(conn_remote, ))

    def update_data(self, measured_direction, target_direction):
        self.conn.send(('update_data', (measured_direction, target_direction)))

    def start(self):
        self.p.start()


if __name__ == '__main__':
    measured_direction = np.random.randint(2, size=(19, 200))
    target_direction = np.random.randint(2, size=(19, 200))

    vis = VisFeatures()
    vis.start()
    vis.update_data(measured_direction, target_direction)
