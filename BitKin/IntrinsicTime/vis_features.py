
import numpy as np
import pygame


class VisFeatures:
    def __init__(self, measured_direction, target_direction):
        self.measured_direction = measured_direction
        self.target_direction = target_direction


    def run(self):
        pygame.init()

        clock = pygame.time.Clock()
        display_size = (800, 600)
        game_display = pygame.display.set_mode(display_size)
        pygame.display.set_caption('Directions')
        display_rect = pygame.Rect((0, 0), display_size)
        pygame.display.update()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == ord('q'):
                        running = False

            game_display.fill(color=(255, 255, 255))

            pygame.draw.rect(game_display, 123, (50, 50, 50, 50))
            pygame.display.update()

            clock.tick(20)

        pygame.quit()


if __name__ == '__main__':
    measured_direction = np.random.randint(1, size=(19, 200))
    target_direction = np.random.randint(1, size=(19, 200))

    vis = VisFeatures(measured_direction, target_direction)
    vis.run()
