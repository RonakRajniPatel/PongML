# Import the pygame library and initialise the game engine
import os
import os.path
import pygame

import screen_capturer
from ball import Ball
from paddle import Paddle


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


pygame.init()

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FRAME_RATE = 60

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

ball = Ball(WHITE, 10, 10)
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
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_x:  # Pressing the x Key will quit the game
                carryOn = False

    # Moving the paddles when the user uses the arrow keys (player A) or "W/S" keys (player B)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        paddleA.move_up(5)
    if keys[pygame.K_s]:
        paddleA.move_down(5)
    if keys[pygame.K_UP]:
        paddleB.move_up(5)
    if keys[pygame.K_DOWN]:
        paddleB.move_down(5)

    # --- Game logic should go here
    all_sprites_list.update()

    # do the intelligent movement
    # ball is heading towards paddleB
    if ball.velocity[0] > 0:
        frames_until_collision = int((690 - ball.rect.x) / ball.velocity[0])
        total_y_to_travel = frames_until_collision * abs(ball.velocity[1])
        bounces = 0
        # ball is travelling up
        if ball.velocity[1] > 0:
            dist_1 = total_y_to_travel - (500 - ball.rect.y)
            bounces = int(dist_1 / 500)
        # ball is travelling down
        else:
            dist_1 = total_y_to_travel - ball.rect.y
            bounces = int(dist_1 / 500)

        remaining_y_to_travel = total_y_to_travel - (bounces * 500)
        predicted_y = 0
        if bounces % 2 == 0:
            if ball.velocity[1] > 0:
                predicted_y = ball.rect.y + remaining_y_to_travel
            else:
                predicted_y = ball.rect.y - remaining_y_to_travel
        else:
            # if the ball is heading up, it'll end up going down
            if ball.velocity[1] > 0:
                predicted_y = ball.rect.y - remaining_y_to_travel
            else:
                predicted_y = ball.rect.y + remaining_y_to_travel
        print(f'frames until collision: {frames_until_collision}')
        print(f'bounces: {bounces}')
        print(predicted_y)
        paddleB.head_to_y(predicted_y, 5)

    # Check if the ball is bouncing against any of the 4 walls:
    if ball.rect.x >= 690:
        scoreA += 1
        ball.go_home()
    if ball.rect.x <= 0:
        scoreB += 1
        ball.go_home()

    if ball.rect.y > 490:
        ball.velocity[1] = -ball.velocity[1]
    if ball.rect.y < 0:
        ball.velocity[1] = -ball.velocity[1]

    # Detect collisions between the ball and the paddles
    if pygame.sprite.collide_mask(ball, paddleA) or pygame.sprite.collide_mask(ball, paddleB):
        ball.bounce()

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

# Once we have exited the main program loop we can stop the game engine:
pygame.quit()
