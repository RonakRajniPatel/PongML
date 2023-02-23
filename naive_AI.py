# predicted_y = 0


def predict_y(ball_velocity, ball_pos):
    frames_until_collision = int((345 - ball_pos.x) / ball_velocity[0])
    total_y_to_travel = frames_until_collision * abs(ball_velocity[1])
    bounces = 0
    # ball is travelling up
    if ball_velocity[1] > 0:
        dist_1 = total_y_to_travel - (250 - ball_pos.y)
        bounces = int(dist_1 / 250)
    # ball is travelling down
    else:
        dist_1 = total_y_to_travel - ball_pos.y
        bounces = int(dist_1 / 250)

    remaining_y_to_travel = total_y_to_travel - (bounces * 250)
    predicted_y = 0
    if bounces % 2 == 0:
        if ball_velocity[1] > 0:
            predicted_y = ball_pos.y + remaining_y_to_travel
        else:
            predicted_y = ball_pos.y - remaining_y_to_travel
    else:
        # if the ball is heading up, it'll end up going down
        if ball_velocity[1] > 0:
            predicted_y = ball_pos.y - remaining_y_to_travel
        else:
            predicted_y = ball_pos.y + remaining_y_to_travel
    # print_summary(frames_until_collision, bounces, predicted_y)
    just_bounced_off_player = False
    return predicted_y


def print_summary(frames, bounce, pred_y):
    print(f'frames until collision: {frames}')
    print(f'bounces: {bounce}')
    print(pred_y)
