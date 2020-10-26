
import pygame


class VisFeatures:
    def __init__(self):
        pass

    def run(self):
        pygame.init()

        display_size = (800, 600)
        game_display = pygame.display.set_mode(display_size)
        pygame.display.update()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()


if __name__ == '__main__':
    vis = VisFeatures()
    vis.run()
