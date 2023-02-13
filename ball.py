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

        self.velocity = [randint(self.SPEED_FACTOR * 4, self.SPEED_FACTOR * 8),
                         randint(self.SPEED_FACTOR * -8, self.SPEED_FACTOR * 8)]

        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()

    def update(self):
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]

    def bounce(self):
        self.velocity[0] = -self.velocity[0]
        self.velocity[1] = randint(self.SPEED_FACTOR * -8, self.SPEED_FACTOR * 8)

    def go_home(self):
        self.rect.x = 345
        self.rect.y = 195
        pygame.time.wait(1000)
        rand_dir = randint(-8, 8)
        while rand_dir == 0:
            rand_dir = randint(-8, 8)
        self.velocity = [self.SPEED_FACTOR * rand_dir, self.SPEED_FACTOR * randint(-8, 8)]
