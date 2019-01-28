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

def run_episode(model):
    """
    The core of training.
    For each movement (until the variable done != True) calculates value function at time T0 and T1,
    based on explored action a0 ~ A0 + exploration.
    Fits critic and actor according to learning rule.
    Saves each step into variable trajectory, and at the end returns full list of steps.
    """

    # initialize variables and reset environment
    trajectory = []
    observation0 = model.env.reset()
    done = False
    while not done:
        if model.env.simulation:
            model.env.render()

        # get current value of value function for observation0
        V0 = model.critic.predict(np.array([observation0]))
        # predict default action
        A0 = model.actor.predict(np.array([observation0]))
        # sample new explored action
        a0 = model.sample(A0[0], model.exploration_factor)
        a0 = [a0]

        # save joint positions in case they need to be reverted. then, make a env.step().
        joint_positions0 = model.env.get_joint_positions()
        observation1, reward, done, info = model.env.step(a0[0])
        # update action to actual action.
        a0 = info["actual_action"]
        #get current value of value function for observation1 and compute delta.
        V1 = model.critic.predict(np.array([observation1]))
        delta = reward + model.gamma * V1 - V0

        # fit critic
        model.critic.fit(np.array([observation0]), [reward + model.gamma * V1], batch_size=1, verbose=0)

        if delta > 0:
            # if delta is positive, fit actor
            model.actor.fit(np.array([observation0]), [a0], batch_size=1, verbose=0)
            observation0 = observation1
        else:
            # otherwise revert joints to position before model.env.step().
            model.env.set_joint_positions(joint_positions0)

        # save and append trajectory.
        step = {"observation0": observation0, "observation1": observation1,
                "V0": V0[0], "V1": V1[0], "A0": A0[0][:], "a0": a0[0][:],
                "reward": reward, "delta": delta[0][0]}
        trajectory.append(step)

    return trajectory


def run_batch(model, batch_size):
    """
    Accepts CACLA model and 'batch size'. Runs number of episodes equal to batch_size.
    Logs the rewards and at the end returns all traversed trajectories.
    """
    trajectories = []
    total_steps = 0

    # run n=batch_size episodes. save trajectories on the way.
    for _ in range(batch_size):
        trajectory = run_episode(model)
        total_steps += len(trajectory)

        trajectories.append(trajectory)

    # call functions for logging
    last_rewards = [trajectory[-1]["reward"] for trajectory in trajectories]
    logger.log({"_MeanReward": np.mean([t["reward"] for trajectory in trajectories for t in trajectory]),
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
    # precompute some values
    rewards = [t["reward"] for trajectory in trajectories for t in trajectory]
    actions_0 = [t["A0"] for trajectory in trajectories for t in trajectory]
    policy_loss = [np.square(np.array(t["a0"]) - np.array(t["A0"])).mean() for trajectory in trajectories for t in trajectory]
    deltas = [np.square(np.array(t["delta"])).mean() for trajectory in trajectories for t in trajectory]
    observations = [t["observation0"] for trajectory in trajectories for t in trajectory]

    # construct dictionary
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


def train(model, n_episodes, batch_size):
    """
    Accepts model (CACLA in our case), number of all episodes, batch size and optional argument to run GUI.
    Trains the actor and critic for number of episodes.
    """

    episode = 0
    best_reward = 0
    while episode < n_episodes:
        # compute trajectories, that will later be used for logging.
        trajectories = run_batch(model, batch_size)
        episode += batch_size

        # save logging data
        d = log_batch_stats(trajectories, episode, model.alpha, model.beta, model.exploration_factor)
        # if this iteration was the best so far, save the model.
        if d["mean_last_reward"] >= best_reward:
            best_reward = d["mean_last_reward"]
            pickle.dump(model, open(logger.path + "/model.pickle", "wb"))
        logger.write(display=True)

        # update learning and exploration rates for the algorithm.
        model.update_lr(model.lr_decay)
        model.update_exploration()

    # at the end of training, save model and close logger.
    pickle.dump(model, open(logger.path + "/model_final.pickle", "wb"))
    logger.close()


def test(model, n):
    """
    tests the model.
    n is the number of locations to test.
    """
    success = 0
    for i in range(n):
        # reset the environment.
        observation = model.env.reset()
        done = False
        # repeat until done (arm reaches the target / 100 steps).
        while not done:
            if model.env.simulation:
                model.env.render()
            # use actor to predict next action.
            action = model.actor.predict(np.array([observation]))

            # make a step.
            observation, reward, done, info = model.env.step(action[0])
            print("iteration:", i, "reward:", reward, "distance:", info["distance"], "done:", done)
            if info["distance"] < 0.01:
                success += 1
        model.env.render()
    print("success rate:", success / n)

# set some global variables that determine log file save location.
env_name = "V-rep_AL5D_no_sim"
now = datetime.utcnow().strftime("%b-%d_%H.%M.%S")  # for unique directories
logger = None

if __name__ == "__main__":
    # initialize the environment (arm) first. 
    # action multiplier is a number by which every perormed action gets multiplied.
    action_multiplier = 0.1
    env = arm.VrepArm(action_multiplier=action_multiplier)

    # comment this when training. change path to point to a saved model.
    """cacla = pickle.load(open("log-files/V-rep_AL5D_no_sim/Jan-27_23.22.56/model_final.pickle", "rb"))
    cacla.env = env
    test(cacla, 1000)
    exit()"""

    # initialize parameters
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    alpha = 0.0008  # learning rate for actor
    beta = 0.0011  # learning rate for critic
    lr_decay = 0.997   # lr decay
    exploration_decay = 0.997   # exploration decay
    gamma = 0.0  # discount factor
    exploration_factor = 0.25

    n_episodes = 10000
    batch_size = 50

    # initialize logger
    logger = Logger(logname=env_name, now=now)

    # run training. when done, reinitialize arm with simulation on to see how it performs in V-rep. run tests.
    cacla = Cacla(env, input_dim, output_dim, alpha, beta, gamma, lr_decay, exploration_decay, exploration_factor)
    train(cacla, n_episodes, batch_size)
    input("Continue?")
    env = arm.VrepArm(simulation=True, action_multiplier=action_multiplier)
    cacla.env = env
    test(cacla, 1000)
