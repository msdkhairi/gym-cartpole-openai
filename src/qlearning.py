import sys, argparse
import numpy as np
import gymnasium as gym


def process_args(args):
    parser = argparse.ArgumentParser(description='Q-Learning')
    parser.add_argument('--test', dest='test_run',
                        type=bool, default=False,
                        help=('If the experiment is a test run set this to True.'
                        ))
    parser.add_argument('--qtable', dest="qtable_path",
                        type=str, default='q_table.npy',
                        help=('Path to load q-table when in test mode. Path to save q-table when in train mode, default=q_table.npy' 
                        ))
    parameters = parser.parse_args(args)
    return parameters




# Define Q-value index function
def q_index(observation, num_buckets, env):
    x, x_dot, theta, theta_dot = observation
    env_x_threshold = env.unwrapped.x_threshold
    x_bins = np.linspace(-env_x_threshold - 1, env_x_threshold + 1, num=num_buckets[0] + 1)
    x_dot_bins = np.linspace(-100, 100, num=num_buckets[1] + 1)
    theta_bins = np.linspace(-np.radians(15), np.radians(15), num=num_buckets[2] + 1)
    theta_dot_bins = np.linspace(-np.radians(50), np.radians(50), num=num_buckets[3] + 1)

    index = np.digitize(x, x_bins) - 1
    index = index * num_buckets[1] + np.digitize(x_dot, x_dot_bins) - 1
    index = index * num_buckets[2] + np.digitize(theta, theta_bins) - 1
    index = index * num_buckets[3] + np.digitize(theta_dot, theta_dot_bins) - 1

    return int(index)


def main(args):

    parameters = process_args(args)

    # Initialize environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    if parameters.test_run:
        q_table = np.load(parameters.qtable_path)
        num_buckets = np.array([1, 1, 8, 4])
        score = 0

        observation, _ = env.reset()
        env._max_episode_steps = 300

        for _ in range(500):
            env.render()
            st = q_index(observation, num_buckets, env)
            action = np.argmax(q_table[st])
            observation, reward, done, info, _ = env.step(action)
            score += reward
            if done:
                break

        env.close()
        print("Test Score =", score)
    else:
        gamma = 0.5
        epsilon = 1.0
        epsilon_decay = 0.998
        epsilon_min = 0.1
        alpha = 0.05

        num_buckets = np.array([1, 1, 8, 4])  # x, x_dot, theta, theta_dot
        q_table = np.zeros((np.prod(num_buckets), env.action_space.n))
        env._max_episode_steps = 500

        episodes = 100
        for i in range(episodes + 1):
            observation, _ = env.reset()

            for _ in range(200):
                st = q_index(observation, num_buckets, env)

                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[st])

                epsilon *= epsilon_decay
                observation, reward, done, info, _ = env.step(action)

                if abs(observation[3]) > np.radians(50):
                    break

                st_new = q_index(observation, num_buckets, env)
                q_table[st, action] = (1 - alpha) * q_table[st, action] + alpha * (
                        reward + gamma * np.amax(q_table[st_new]))

                if done:
                    break

            if i % (episodes / 100) == 0:
                print(f'{int(i / (episodes / 100))}% completed')

        score = 0
        observation, _ = env.reset()

        for _ in range(500):
            env.render()
            st = q_index(observation, num_buckets, env)
            action = np.argmax(q_table[st])
            observation, reward, done, info, _ = env.step(action)
            score += reward
            if done:
                break

        env.close()
        np.save('q_table.npy', q_table)
        print("Training Complete - Final Score =", score)
        print(f"Q-Table Saved to {parameters.qtable_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
    