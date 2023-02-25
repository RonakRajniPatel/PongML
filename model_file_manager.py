import os.path

FOLDER_NAME = "model_data"


def initialize_model_dir():
    if not os.path.isdir(FOLDER_NAME):
        os.mkdir(FOLDER_NAME)


def store_epsilon(new_epsilon):
    full_path = os.path.join(FOLDER_NAME, "epsilon_value.txt")
    try:
        epsilon_value = open(full_path, "r+")
        epsilon_value.seek(0)
        epsilon_value.truncate()
        epsilon_value.write(f"{new_epsilon}")
        epsilon_value.close()
        print(f"Epsilon value of {new_epsilon} stored successfully!")
    except IOError:
        print("Epsilon Value is not stored! Initializing")
        epsilon_value = open(full_path, "w")
        epsilon_value.write(f"{new_epsilon}")
        print(f"Epsilon value of {new_epsilon} stored successfully!")
        epsilon_value.close()


def get_epsilon():
    full_path = os.path.join(FOLDER_NAME, "epsilon_value.txt")
    try:
        epsilon_value = open(full_path, "r+")
        epsilon_data = epsilon_value.read()
        epsilon = float(epsilon_data)
        epsilon_value.close()
        print(f"Epsilon value of {epsilon} retrieved from file.")
        return epsilon
    except IOError:
        print("Epsilon Value is not stored! Initializing")
        epsilon_value = open(full_path, "w")
        epsilon_value.write(f"{1.0}")
        print("Epsilon file created. Epsilon value initialized to 1.0")
        epsilon_value.close()
        return 1.0


def store_episodes(new_episodes):
    full_path = os.path.join(FOLDER_NAME, "episode_value.txt")
    try:
        episode_value = open(full_path, "r+")
        episode_value.seek(0)
        episode_value.truncate()
        episode_value.write(f"{new_episodes}")
        episode_value.close()
        print(f"Episode of {new_episodes} stored successfully!")
    except IOError:
        print("Episode Value is not stored! Initializing")
        episode_value = open(full_path, "w")
        episode_value.write(f"{new_episodes}")
        print(f"Episode value of {new_episodes} stored successfully!")
        episode_value.close()


def get_episodes():
    full_path = os.path.join(FOLDER_NAME, "episode_value.txt")
    try:
        episode_value = open(full_path, "r+")
        episode_data = episode_value.read()
        episodes = int(episode_data)
        episode_value.close()
        print(f"Episode value of {episodes} retrieved from file.")
        return episodes
    except IOError:
        print("Episode Value is not stored! Initializing")
        episode_value = open(full_path, "w")
        episode_value.write(f"{0}")
        print("Episode file created. Episode value initialized to 0")
        episode_value.close()
        return 1


def store_frames(new_frames):
    full_path = os.path.join(FOLDER_NAME, "frame_count_value.txt")
    try:
        frame_value = open(full_path, "r+")
        frame_value.seek(0)
        frame_value.truncate()
        frame_value.write(f"{new_frames}")
        frame_value.close()
        print(f"Frame count of {new_frames} stored successfully!")
    except IOError:
        print("Frame count Value is not stored! Initializing")
        frame_value = open(full_path, "w")
        frame_value.write(f"{new_frames}")
        print(f"Frame count value of {new_frames} stored successfully!")
        frame_value.close()


def get_frames():
    full_path = os.path.join(FOLDER_NAME, "frame_count_value.txt")
    try:
        frame_value = open(full_path, "r+")
        frame_data = frame_value.read()
        frames = int(frame_data)
        frame_value.close()
        print(f"Frame count value of {frames} retrieved from file.")
        return frames
    except IOError:
        print("Frame count Value is not stored! Initializing")
        frame_value = open(full_path, "w")
        frame_value.write(f"{0}")
        print("Frame count file created. Frame value initialized to 0")
        frame_value.close()
        return 1


def store_all(new_epsilon, new_episodes, new_frames):
    store_epsilon(new_epsilon)
    store_episodes(new_episodes)
    store_frames(new_frames)
