# Import the pygame library and initialise the game engine
import os
import os.path

import pygame
import numpy as np
import random

import naive_AI
import screen_capturer
from ball import Ball
from paddle import Paddle

from IPython.display import clear_output

num_episodes = 1000
close_game = False


def init_picture_dir():
    directory = 'pics'
    picture_index = 0
    if os.path.exists(directory):
        for path in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, path)):
                picture_index += 1
        print(picture_index)
    else:
        os.mkdir(directory)
        picture_index = 0
    return picture_index


def reset_game():
    return False


for episode in range(0, num_episodes):
    pygame.init()

    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    SPEED_FACTOR = 1
    FRAME_RATE = 60
    PLAYER_BASE_MOVEMENT_SPEED = 5
    predicted_y = 0
    just_bounced = False

    # Open a new window
    size = (700, 500)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Pong")

    # paddleA is the "player", aka the human player in Pong
    # eventually, this will become the reinforcement learner
    paddleA = Paddle(WHITE, 10, 100)
    paddleA.rect.x = 20
    paddleA.rect.y = 200

    # paddleB is the "computer", aka the traditional opponent in Pong
    paddleB = Paddle(WHITE, 10, 100)
    paddleB.rect.x = 670
    paddleB.rect.y = 200

    ball = Ball(WHITE, 10, 10, SPEED_FACTOR)
    ball.rect.x = 345
    ball.rect.y = 195

    # This will be a list that will contain all the sprites we intend to use in our game.
    all_sprites_list = pygame.sprite.Group()

    # Add the paddles to the list of sprites
    all_sprites_list.add(paddleA)
    all_sprites_list.add(paddleB)
    all_sprites_list.add(ball)

    # The loop will carry on until the user exits the game (e.g. clicks the close button).
    carryOn = True

    # The clock will be used to control how fast the screen updates
    clock = pygame.time.Clock()

    # Initialise player scores
    scoreA = 0
    scoreB = 0

    PICTURE_INDEX = init_picture_dir()
    # screen_capturer.capture(PICTURE_INDEX)
    print(PICTURE_INDEX)

    # -------- Main Program Loop -----------
    while carryOn:
        # --- Main event loop
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                carryOn = False  # Flag that we are done so we exit this loop
                close_game = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:  # Pressing the x Key will quit the game
                    carryOn = False
                    close_game = True

        # Moving the paddles when the user uses the arrow keys (player A) or "W/S" keys (player B)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            paddleA.move_up(SPEED_FACTOR * PLAYER_BASE_MOVEMENT_SPEED)
        if keys[pygame.K_s]:
            paddleA.move_down(SPEED_FACTOR * PLAYER_BASE_MOVEMENT_SPEED)
        # if keys[pygame.K_UP]:
        #     paddleB.move_up(5)
        # if keys[pygame.K_DOWN]:
        #     paddleB.move_down(5)
        paddleB.head_to_y(predicted_y, SPEED_FACTOR * PLAYER_BASE_MOVEMENT_SPEED)

        # --- Game logic should go here
        all_sprites_list.update()

        # do the intelligent movement
        # ball is heading towards paddleB
        if just_bounced:
            if ball.velocity[0] > 0:
                predicted_y = naive_AI.predict_y(ball.velocity, ball.rect)
            just_bounced = False

        # Check if the ball is bouncing against any of the 4 walls:
        if ball.rect.x >= 690:
            scoreA += 1
            ball.go_home()
            just_bounced = True
        if ball.rect.x <= 0:
            scoreB += 1
            ball.go_home()
            just_bounced = True

        if ball.rect.y > 490:
            ball.velocity[1] = -ball.velocity[1]
            just_bounced = True
        if ball.rect.y < 0:
            ball.velocity[1] = -ball.velocity[1]
            just_bounced = True

        # Detect collisions between the ball and the paddles
        if pygame.sprite.collide_mask(ball, paddleA) or pygame.sprite.collide_mask(ball, paddleB):

            ball.bounce()
            just_bounced = True

        # --- Drawing code should go here
        # First, clear the screen to black.
        screen.fill(BLACK)
        # Draw the net
        pygame.draw.line(screen, WHITE, [349, 0], [349, 500], 5)

        # Now let's draw all the sprites in one go. (For now we only have 2 sprites!)
        all_sprites_list.draw(screen)

        # Display scores:
        font = pygame.font.Font(None, 74)
        text = font.render(str(scoreA), 1, WHITE)
        screen.blit(text, (250, 10))
        text = font.render(str(scoreB), 1, WHITE)
        screen.blit(text, (420, 10))

        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        # --- Limit to 60 frames per second
        clock.tick(FRAME_RATE)

        # if either player has a score of 10 or more, reset the game
        if scoreA >= 4 or scoreB >= 4:
            carryOn = reset_game()
    if close_game:
        break

# Once we have exited the main program loop we can stop the game engine:
pygame.quit()
