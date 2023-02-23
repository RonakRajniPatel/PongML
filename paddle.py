import pygame

BLACK = (0, 0, 0)


class Paddle(pygame.sprite.Sprite):
    # This class represents a paddle. It derives from the "Sprite" class in Pygame.

    def __init__(self, color, width, height):
        # Call the parent class (Sprite) constructor
        super().__init__()

        # Pass in the color of the paddle, its width and height.
        # Set the background color and set it to be transparent
        self.image = pygame.Surface([width, height])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)

        # Draw the paddle (a rectangle!)
        pygame.draw.rect(self.image, color, [0, 0, width, height])

        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()

    # if this paddle is an AI, do this method in the main loop
    #def
        # check where the ball is and see where I need to move

    def move_up(self, pixels):
        self.rect.y -= pixels
        # Check that you are not going too far (off the screen)
        if self.rect.y < 0:
            self.rect.y = 0

    def move_down(self, pixels):
        self.rect.y += pixels
        # Check that you are not going too far (off the screen)
        if self.rect.y > 200:
            self.rect.y = 200

    def stay_here(self):
        return

    def head_to_y(self, destination, pixels):
        if self.rect.y < 0:
            self.rect.y = 0
        elif self.rect.y > 200:
            self.rect.y = 200
        if not ((self.rect.y + 25) == destination):
            if destination > (self.rect.y + 25):
                self.rect.y += pixels
            else:
                self.rect.y -= pixels
