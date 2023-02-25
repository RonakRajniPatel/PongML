import pygame
from random import randint

BLACK = (0, 0, 0)
SPEED_FACTOR = 0


class Ball(pygame.sprite.Sprite):
    # This class represents a ball. It derives from the "Sprite" class in Pygame.

    def __init__(self, color, width, height, speed_factor):
        # Call the parent class (Sprite) constructor
        super().__init__()

        # Pass in the color of the ball, its width and height.
        # Set the background color and set it to be transparent
        self.SPEED_FACTOR = speed_factor
        self.image = pygame.Surface([width, height])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)

        # Draw the ball (a rectangle!)
        pygame.draw.rect(self.image, color, [0, 0, width, height])

        self.velocity = [randint(self.SPEED_FACTOR * 3, self.SPEED_FACTOR * 4),
                         randint(self.SPEED_FACTOR * -4, self.SPEED_FACTOR * 4)]

        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()

    def update(self):
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]

    def bounce_off_wall(self):
        # this if/else block addresses the issue where the ball would get stuck at the top or bottom
        # of the screen and awkwardly slide across, making the game progress very slowly
        if self.rect.y > 200: # ball is anywhere near the top of the screen
            self.velocity[1] = -abs(self.velocity[1])
        else: # ball is more likely near the bottom of the screen
            self.velocity[1] = abs(self.velocity[1])

    def bounce_off_paddle(self, player_paddle):
        if player_paddle:
            self.velocity[0] = abs(self.velocity[0])
        else:
            self.velocity[0] = -abs(self.velocity[0])
        self.velocity[1] = randint(self.SPEED_FACTOR * -4, self.SPEED_FACTOR * 4)

    def go_home(self):
        self.rect.x = 172
        self.rect.y = 97
        pygame.time.wait(800)
        rand_dir = randint(-4, 4)
        while rand_dir == 0:
            rand_dir = randint(-4, 4)
        self.velocity = [self.SPEED_FACTOR * rand_dir, self.SPEED_FACTOR * randint(-4, 4)]
