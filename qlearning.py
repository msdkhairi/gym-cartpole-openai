import numpy as np
import gym
import math
import sys


env = gym.make('CartPole-v0')


def q_index(observation, num_buckets):
    total_index = np.prod(num_buckets)
    x, x_dot, theta, theta_dot = observation
    index = 0

    x_thr = env.env.x_threshold + 1
    x_dot_thr = 100
    theta_thr = math.radians(15)
    theta_dot_thr = math.radians(50)

    x_bins = np.linspace(-x_thr, x_thr, num=num_buckets[0] + 1)
    index += (np.digitize(x, x_bins) - 1) * total_index / num_buckets[0]

    x_dot_bins = np.linspace(-x_dot_thr, x_dot_thr, num=num_buckets[1] + 1)
    index += (np.digitize(x_dot, x_dot_bins) - 1) * total_index / num_buckets[1] / num_buckets[0]

    theta_bins = np.linspace(-theta_thr, theta_thr, num=num_buckets[2] + 1)
    index += (np.digitize(theta, theta_bins) - 1) * total_index / num_buckets[2] / num_buckets[1] / num_buckets[0]

    theta_dot_bins = np.linspace(-theta_dot_thr, theta_dot_thr, num=num_buckets[3] + 1)
    index += (np.digitize(theta_dot, theta_dot_bins) - 1) * total_index / num_buckets[3] / num_buckets[2] / num_buckets[
        1] / num_buckets[0]

    return np.int(index)



if sys.argv[-1] == 'test':
    q_table = np.load('src/q_table.npy')
    num_buckets = np.array([1,1,8,4])
    score = 0
    observation = env.reset()
    env._max_episode_steps = 200;
    for _ in range(500):
        # env.render()
        st = q_index(observation,num_buckets)
        action = np.argmax(q_table[st])
        observation, reward, done, info = env.step(action)
        score += reward
        if done:
            break
    # env.close()
    print(score)

else:
    gamma = 0.6
    epsilon = 1.0
    epsilon_decay = 0.998
    epsilon_min = 0.1
    alpha = 0.1

    num_buckets = np.array([1, 1, 8, 4])  # x, x_dot, theta, theta_dot
    q_table = np.zeros((np.prod(num_buckets), env.action_space.n))
    env._max_episode_steps = 500

    episodes = 100
    for i in range(episodes + 1):
        observation = env.reset()
        for _ in range(200):
            st = q_index(observation, num_buckets)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[st])
            epsilon *= epsilon_decay
            observation, reward, done, info = env.step(action)
            if abs(observation[3]) > math.radians(50):
                break
            st_new = q_index(observation, num_buckets)
            q_table[st, action] = (1 - alpha) * q_table[st, action] + alpha * (
                    reward + gamma * np.amax(q_table[st_new]))
            if done:
                break
        if i % (episodes / 100) == 0:
            print('{}% completed'.format(i / (episodes / 100)))

    score = 0
    observation = env.reset()
    for _ in range(500):
        # env.render()
        st = q_index(observation, num_buckets)
        action = np.argmax(q_table[st])
        observation, reward, done, info = env.step(action)
        score += reward
        if done:
            break
    # env.close()
    print('Score=',score)


