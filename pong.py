# Import the pygame library and initialise the game engine
import os
import os.path
import pygame

import model_file_manager
import naive_AI
import screen_capturer
from ball import Ball
from paddle import Paddle
import naive_AI
import numpy as np
import tensorflow as tf
import DQLearner
import random
from IPython.display import clear_output
from skimage.transform import resize
from skimage.color import rgb2gray


# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
#epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000
model_file_manager.initialize_model_dir()
model_filepath = os.path.join("model_data", "model")
model_target_filepath = os.path.join("model_data", "model_target")
# save the learner whenever the game is exited
SAVE_LEARNER = True
RESTART_LEARNING = False

if RESTART_LEARNING:
    epsilon = 1.0
    episode_count = 0
    frame_count = 0
    model_file_manager.store_epsilon(epsilon)
    model_file_manager.store_episodes(episode_count)
    model_file_manager.store_frames(frame_count)
    # The first model makes the predictions for Q-values which are used to
    # make a action.
    model = DQLearner.create_q_model()
    print(model.summary())
    # Build a target model for the prediction of future rewards.
    # The weights of a target model get updated every 10000 steps thus when the
    # loss between the Q-values is calculated the target Q-value is stable.
    model_target = DQLearner.create_q_model()
else:
    epsilon = model_file_manager.get_epsilon()
    episode_count = model_file_manager.get_episodes()
    frame_count = model_file_manager.get_frames()
    if os.path.isfile(model_filepath) and os.path.isfile(model_target_filepath):
        model = tf.keras.models.load_model(model_filepath)
        model_target = tf.keras.models.load_model(model_target_filepath)
    else:
        model = DQLearner.create_q_model()
        model_target = DQLearner.create_q_model()

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0

#episode_count = 0

#frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 16000
# Train the model after 4 actions
update_after_actions = 8
# How often to update the target network
update_target_network = 10000
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
# Speed to multiply all movements by
SPEED_FACTOR = 1
# Maximum number of allowed frames per second
FRAME_RATE = 120
# How many pixels per frame the player can move
PLAYER_BASE_MOVEMENT_SPEED = 3

num_episodes = 10000
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


for episode in range(0, (num_episodes - episode_count)):
    pygame.init()

    predicted_y = 0
    just_bounced = False
    frame_count_episode = 0
    state = np.zeros(shape=(350, 250))
    episode_reward = 0
    playerA_times_paddled = 0

    # Open a new window
    size = (350, 250)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Pong")

    # paddleA is the "player", aka the human player in Pong
    # eventually, this will become the reinforcement learner
    paddleA = Paddle(WHITE, 5, 50)
    paddleA.rect.x = 10
    paddleA.rect.y = 100

    # paddleB is the "computer", aka the traditional opponent in Pong
    paddleB = Paddle(WHITE, 5, 50)
    paddleB.rect.x = 335
    paddleB.rect.y = 100

    ball = Ball(WHITE, 5, 5, SPEED_FACTOR)
    ball.rect.x = 172
    ball.rect.y = 97

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

    #PICTURE_INDEX = init_picture_dir()
    # screen_capturer.capture(PICTURE_INDEX)
    #print(PICTURE_INDEX)

    # -------- Main Program Loop -----------
    while carryOn:
        frame_count += 1
        frame_count_episode += 1
        reward = 0

        # --- Main event loop
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                carryOn = False  # Flag that we are done so we exit this loop
                close_game = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:  # Pressing the x Key will quit the game
                    carryOn = False
                    close_game = True
                # pressing the k key ends the current episode (if the ball gets stuck or whatever)
                elif event.key == pygame.K_k:
                    carryOn = reset_game()

        paddleB.head_to_y(predicted_y, SPEED_FACTOR * PLAYER_BASE_MOVEMENT_SPEED)

        # --- Game logic should go here
        all_sprites_list.update()

        # TODO: BEGIN DEEP Q LEARNER SECTION

        # Use epsilon-greedy for exploration
        if frame_count_episode < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(DQLearner.num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        #state_next, reward, done, _ = env.step(action)

        ###
        # Moving the paddles when the user uses the arrow keys (player A) or "W/S" keys (player B)
        #print(f'action = {action}')
        keys = pygame.key.get_pressed()
        if action == 0:
            paddleA.move_up(SPEED_FACTOR * PLAYER_BASE_MOVEMENT_SPEED)
        elif action == 1:
            paddleA.move_down(SPEED_FACTOR * PLAYER_BASE_MOVEMENT_SPEED)
        else:
            # action = 3
            paddleA.stay_here()
        ###

        # do the intelligent movement
        # ball is heading towards paddleB
        if just_bounced:
            if ball.velocity[0] > 0:
                predicted_y = naive_AI.predict_y(ball.velocity, ball.rect)
            just_bounced = False

        # Check if the ball is bouncing against any of the 4 walls:
        if ball.rect.x >= 345:
            scoreA += 1
            reward += .25
            ball.go_home()
            predicted_y = naive_AI.predict_y(ball.velocity, ball.rect)
            just_bounced = True
        if ball.rect.x <= 5:
            scoreB += 1
            reward -= .75
            ball.go_home()
            predicted_y = naive_AI.predict_y(ball.velocity, ball.rect)
            just_bounced = True

        if ball.rect.y > 245:
            ball.velocity[1] = -ball.velocity[1]
            just_bounced = True
        if ball.rect.y < 0:
            ball.velocity[1] = -ball.velocity[1]
            just_bounced = True

        player_dist_to_ball = abs(paddleA.rect.y - ball.rect.y + 23) + .2
        #print(f'dist = {player_dist_to_ball}')
        #reward = scoreA
        #reward -= scoreB
        dist_reward = min(2, (4 * pow(player_dist_to_ball, -1)))
        reward += dist_reward
        #reward += 2/(player_dist_to_ball + 1)

        # Detect collisions between the ball and the paddles
        if pygame.sprite.collide_mask(ball, paddleA) or pygame.sprite.collide_mask(ball, paddleB):
            if pygame.sprite.collide_mask(ball, paddleA):
                ball.bounce_off_paddle(True)
                playerA_times_paddled += 1
                reward += .75
            else:
                # ball bounced off the opponent (right side) paddle
                ball.bounce_off_paddle(False)
            just_bounced = True

        #print(f'reward = {reward}')

        #state_next = np.array(state_next)
# TODO: THIS IS WHERE THE GAME SCREEN ARRAY IS TAKEN
        # references the array of pixels on the screen (faster)
        state_next = pygame.surfarray.pixels3d(pygame.display.get_surface())

        # copies the array of pixels on the screen
        # state_next = pygame.surfarray.array2d(pygame.display.get_surface())
        state_next = rgb2gray(state_next)
        # state_next = resize(state_next, (175, 125))
        #state_next.swapaxes(0, 1)

        # if either player has a score of 10 or more, reset the game
        if scoreA >= 5 or scoreB >= 5 or frame_count_episode >= 10000:
            if scoreB >= 5:
                reward = -10 + scoreA
            elif scoreA >= 5:
                reward = 10 - scoreB
            carryOn = reset_game()

        episode_reward += reward



        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(carryOn)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count_episode % update_after_actions == 0 and len(done_history) > batch_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            #print(state_history.shape)
            #print("performing sampling logic!")
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample, verbose=0)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, DQLearner.num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = DQLearner.loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            DQLearner.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count_episode % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

            # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        # TODO: END DEEP Q LEARNER SECTION

        # --- Drawing code should go here
        # First, clear the screen to black.
        screen.fill(BLACK)
        # Draw the net
        pygame.draw.line(screen, WHITE, [174, 0], [174, 250], 2)

        # Now let's draw all the sprites in one go. (For now we only have 2 sprites!)
        all_sprites_list.draw(screen)

        # Display scores:
        font = pygame.font.Font(None, 37)
        text = font.render(str(scoreA), 1, WHITE)
        screen.blit(text, (125, 5))
        text = font.render(str(scoreB), 1, WHITE)
        screen.blit(text, (210, 5))

        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        # --- Limit to 60 frames per second
        clock.tick(FRAME_RATE)

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    print(f'Episode {episode_count} is over.')
    print(f'Final Score was {scoreA} - {scoreB}')
    print(f'Learner hit the ball {playerA_times_paddled} times.')

    episode_count += 1

    # save the model every 100th episode
    if episode_count % 100 == 0 and SAVE_LEARNER:
        model.save(model_filepath, save_format="h5", )
        model_target.save(model_target_filepath, save_format="h5")
        model_file_manager.store_epsilon(epsilon)
        model_file_manager.store_episodes(episode_count)
        model_file_manager.store_frames(frame_count)

    if running_reward > 1000:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        if SAVE_LEARNER:
            model.save(model_filepath, save_format="h5",)
            model_target.save(model_target_filepath, save_format="h5")
            model_file_manager.store_epsilon(epsilon)
            model_file_manager.store_episodes(episode_count)
            model_file_manager.store_frames(frame_count)
        break

    if close_game:
        if SAVE_LEARNER:
            model.save(model_filepath, save_format="h5")
            model_target.save(model_target_filepath, save_format="h5")
            model_file_manager.store_epsilon(epsilon)
            model_file_manager.store_episodes(episode_count - 1)
            model_file_manager.store_frames(frame_count)
        break

# Once we have exited the main program loop we can stop the game engine:
pygame.quit()
