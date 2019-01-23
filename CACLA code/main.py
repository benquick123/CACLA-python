import gym
import numpy as np
import time
from cacla import Cacla
from utils import Logger
from datetime import datetime
import pickle
import copy
import vrep_arm3 as arm
import vrep
import serial

env_name = "V-rep_AL5D_no_sim"
now = datetime.utcnow().strftime("%b-%d_%H.%M.%S")  # for unique directories
logger = None


def run_episode(model, episode, animate=False):
    """
    The core of data collection.
    For each movement (until the variable done == True) calculates value function at time T0 and T1,
    based on explored action a0 ~ A0 + exploration.
    Fits critic and actor according to learning rule.
    Saves each step into variable trajectory, and at the end returns full list of steps.
    """
    done = False

    trajectory = []
    observation0 = model.env.reset()
    # iteration_n = 0
    # print("RESET observation", observation0)
    # scale, offset = scaler.get()
    while not done:
        if model.env.simulation:
            model.env.render()

        V0 = model.critic.predict(np.array([observation0]))
        A0 = model.actor.predict(np.array([observation0]))

        joint_positions0 = model.env.get_joint_positions()
        # print("BEFORE STEP joint positions", joint_positions0.tolist())

        a0 = model.sample(A0[0], model.exploration_factor)
        a0 = [a0]
        # print("EXPLORING ACTION", a0)

        # env_state0 = copy.deepcopy(model.env)
        observation1, reward, done, info = model.env.step(a0[0])
        a0 = info["actual_action"]
        # print("AFTER STEP observation, reward", observation1.tolist(), reward)
        V1 = model.critic.predict(np.array([observation1]))
        delta = reward + model.gamma * V1 - V0
        # print("DELTA", delta)

        # fit critic
        model.critic.fit(np.array([observation0]), [reward + model.gamma * V1], batch_size=1, verbose=0)
        # print("FITTING CRITIC")

        if delta > 0:
            # if delta is positive, fit actor
            model.actor.fit(np.array([observation0]), [a0], batch_size=1, verbose=0)
            observation0 = observation1
            # print("FITTING ACTOR; SEE IF OBSERVATION IS CHANGED")
        else:
            # otherwise set things to how they were before model.env.step().
            model.env.set_joint_positions(joint_positions0)
            # print("OBSERVATION SHOULD BE THE SAME AS BEFORE")
            # model.env = env_state0

        step = {"observation0": observation0, "observation1": observation1,
                # "observation_unscaled": observation_unscaled,
                "V0": V0[0], "V1": V1[0], "A0": A0[0][:], "a0": a0[0][:],
                "reward": reward, "delta": delta[0][0]}
        trajectory.append(step)
        # print("OBSERVATION AT END", observation0.tolist())
        # if iteration_n >= 50:
        #     break
        # iteration_n += 1

    return trajectory


def run_batch(model, batch_size, episode, animate=False):
    """
    Accepts CACLA model, scaler and batch size. Runs number of episodes equal to batch_size.
    Logs the rewards and at the end returns all traversed trajectories.
    """
    trajectories = []
    total_steps = 0

    for _ in range(batch_size):
        trajectory = run_episode(model, episode, animate=animate)
        total_steps += len(trajectory)

        trajectories.append(trajectory)

    last_rewards = [trajectory[-1]["reward"] for trajectory in trajectories]
    logger.log({"_MeanReward": np.mean([t["reward"] for trajectory in trajectories for t in trajectory]),
                # "_MeanReward": np.mean([np.sum([t["reward"] for t in trajectory]) for trajectory in trajectories]),
                "Steps": total_steps,
                "mean_last_reward": np.mean(last_rewards),
                "_std_last_reward": np.std(last_rewards),
                "_min_last_reward": np.min(last_rewards),
                "_max_last_reward": np.max(last_rewards)})
    return trajectories


def log_batch_stats(trajectories, episode, alpha, beta, exploration_factor):
    """
    Creates dictionary with values to log.
    return dictionary of those values, if they are to be used somewhere else.
    """
    rewards = [t["reward"] for trajectory in trajectories for t in trajectory]
    actions_0 = [t["A0"] for trajectory in trajectories for t in trajectory]
    policy_loss = [np.square(np.array(t["a0"]) - np.array(t["A0"])).mean() for trajectory in trajectories for t in trajectory]
    deltas = [np.square(np.array(t["delta"])).mean() for trajectory in trajectories for t in trajectory]
    observations = [t["observation0"] for trajectory in trajectories for t in trajectory]

    d = {"_min_reward": np.min(rewards),
         "_max_reward": np.max(rewards),
         "_mean_reward": np.mean(rewards),
         "_std_reward": np.std(rewards),
         "_min_action": np.min(actions_0),
         "_max_action": np.max(actions_0),
         "_mean_action": np.mean(actions_0),
         "_std_action": np.std(actions_0),
         "_min_observation": np.min(observations),
         "_max_observation": np.max(observations),
         "_mean_observation": np.mean(observations),
         "_std_observations": np.std(observations),
         "_min_value_loss": np.min(deltas),
         "_max_value_loss": np.max(deltas),
         "mean_value_loss": np.mean(deltas),
         "_std_value_loss": np.std(deltas),
         "_min_policy_loss": np.min(policy_loss),
         "_max_policy_loss": np.max(policy_loss),
         "mean_policy_loss": np.mean(policy_loss),
         "_std_policy_loss": np.std(policy_loss),
         "policy_lr": alpha,
         "value_lr": beta,
         "exploration_factor": exploration_factor,
         "_episode": episode}

    logger.log(d)
    return d


def train(model, n_episodes, batch_size, animate=False):
    """
    Accepts model (CACLA in our case), number of all episodes, batch size and optional argument to run GUI.
    Trains the actor and critic for number of episodes.
    """

    episode = 0
    best_reward = 0
    while episode < n_episodes:
        # compute trajectories, that CACLA will use to train critic and actor.
        trajectories = run_batch(model, batch_size, episode, animate)
        episode += batch_size

        # save logging data
        d = log_batch_stats(trajectories, episode, model.alpha, model.beta, model.exploration_factor)
        if d["_mean_reward"] >= best_reward:
            best_reward = d["_mean_reward"]
            pickle.dump(model, open(logger.path + "/model.pickle", "wb"))
        logger.write(display=True)

        # update learning and exploration rates for the algorithm.
        model.update_lr(model.lr_decay)
        # if model.exploration_factor < 0.04:
        # exploration_decay = (n_episodes - episode) / (n_episodes - episode + batch_size)
        model.update_exploration()
        # model.update_exploration()

    pickle.dump(model, open(logger.path + "/model_final.pickle", "wb"))
    logger.close()


def test(model, n):
    # model.env = gym.make(env_name)
    success = 0
    for i in range(n):
        observation = model.env.reset()
        done = False
        while not done:
            if model.env.simulation:
                model.env.render()
            action = model.actor.predict(np.array([observation]))

            observation, reward, done, info = model.env.step(action[0])
            print("iteration:", i, "reward:", reward, "distance:", info["distance"], "done:", done)
            if info["distance"] < 0.01:
                success += 1
        model.env.render()
        time.sleep(3)
        print(model.env.get_tip_position())
        _, tip = vrep.simxGetObjectHandle(model.env.clientID, 'AL5D_tip', vrep.simx_opmode_blocking)
        _, real_tip = vrep.simxGetObjectPosition(model.env.clientID, tip, -1, vrep.simx_opmode_blocking)
        print(real_tip)
    print("success rate:", success / n)


"""def real_test(model):
    # new_restrictions = [[-90, 90], [-60, 0], [0, 100], [-90, 45], [-90, 90]]
    new_values = [[500, 2500], [1000, 1500], [950, 2000], [600, 1900], [600, 2400]]
    additional_restrictions = [[-0.5, 0.5], [-2/3, 0.0], [0.0, 0.625], [-1, 0.5], [-0.5, 0.5]]
    conn = serial.Serial("COM3", 115200)
    cr = bytes("\r", "ascii")

    def generate_action(action):
        speed = "S500"
        for rest, a in zip(additional_restrictions, action[0]):
            print(a)
            if a < rest[0]:
                a = rest[0]
            elif a > rest[1]:
                a = rest[1]
            

    # initialize
    init = bytes("#0 P1500 #1 P1500 #2 P1500 #3 P1500 #4 P1500", "ascii")
    conn.write(init + cr)

    done = False
    observation = cacla.env.reset()
    observation[0] = 0.15
    observation[1] = 0.2
    observation[2] = 0.02
    for s in range(50):
        action = model.actor.predict(np.array([observation]))
        model.env.step(action[0])
        serial_action = generate_action(action)
        exit()
"""


if __name__ == "__main__":
    action_multiplier = 0.1
    env = arm.VrepArm(simulation=True, action_multiplier=action_multiplier)
    cacla = pickle.load(open("log-files/V-rep_AL5D_no_sim/Jan-22_15.45.30_best_sim/model_final.pickle", "rb"))
    cacla.env = env
    test(cacla, 20)
    exit()

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    alpha = 0.0007  # learning rate for actor
    beta = 0.001  # learning rate for critic
    lr_decay = 0.997   # lr decay
    exploration_decay = 0.997   # exploration decay
    gamma = 0.0  # discount factor
    exploration_factor = 0.15

    n_episodes = 20000
    batch_size = 50

    logger = Logger(logname=env_name, now=now)

    cacla = Cacla(env, input_dim, output_dim, alpha, beta, gamma, lr_decay, exploration_decay, exploration_factor)
    train(cacla, n_episodes, batch_size)
    input("Continue?")
    env = arm.VrepArm(simulation=True, action_multiplier=action_multiplier)
    cacla.env = env
    test(cacla, 10)
