# predicted_y = 0


def predict_y(ball_velocity, ball_pos):
    frames_until_collision = int((690 - ball_pos.x) / ball_velocity[0])
    total_y_to_travel = frames_until_collision * abs(ball_velocity[1])
    bounces = 0
    # ball is travelling up
    if ball_velocity[1] > 0:
        dist_1 = total_y_to_travel - (500 - ball_pos.y)
        bounces = int(dist_1 / 500)
    # ball is travelling down
    else:
        dist_1 = total_y_to_travel - ball_pos.y
        bounces = int(dist_1 / 500)

    remaining_y_to_travel = total_y_to_travel - (bounces * 500)
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
    print(f'frames until collision: {frames_until_collision}')
    print(f'bounces: {bounces}')
    print(predicted_y)
    just_bounced_off_player = False
    return predicted_y
