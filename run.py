import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dqn.ExperienceReplayMemory import ExperienceReplayMemory
from environments.GeneralOpenAIEnvironment import GeneralOpenAIEnvironment
from dqn.DDQNAgent import DDQNAgent
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy

name = "MountainCar-v0"
run_folder = "./run/"

epochs = 10000
steps_per_epoch = 1000
render = True
monitor = False

discount = 0.99
learning_rate = 0.00025

replay_size = 10000
sample_size = 64
copy_offset = 100
structure = [32, 32]

starter_exploration_rate = 1.0
decay_exploration_steps = 1000000
end_exploration_rate = 0.1

seed = 4

# create the reward graph
rewards = np.zeros(epochs)

graph = tf.Graph()
with graph.as_default():
    with tf.Session(graph=graph) as sess:

        # set seed
        tf.set_random_seed(seed)

        # build environment
        env = GeneralOpenAIEnvironment(name, run_folder, render, monitor)
        observation_space = env.env.observation_space
        action_space = env.env.action_space

        # create the memory and the agent
        memory = ExperienceReplayMemory(replay_size, sample_size, observation_space, action_space)
        agent = DDQNAgent(env.env.observation_space, env.env.action_space, memory, structure, copy_offset)

        # create linear decay learning rate
        exploration_decay = tf.train.polynomial_decay(starter_exploration_rate, agent.global_step,
                                                      decay_exploration_steps, end_exploration_rate,
                                                      power=1)
        # obtain action and state spaces
        exploration_strategy = EpsilonGreedyPolicy(action_space, exploration_decay)

        # build the final graph
        current_observation = env.current_observation
        action = agent.create_action_graph(env.current_observation, exploration_strategy)
        next_observation, reward, done = env.perform_step_op(action)


        error, minimizer = agent.create_learn_graph(current_observation, next_observation,
                                                    action, reward, discount, learning_rate, done)

        # execute the minimizer before "applying" the action
        with tf.control_dependencies([minimizer]):
            one_step = env.apply_op()

        # init the graph
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # reset the environment
        env_res_op = env.reset_op()
        sess.run(env_res_op)

        # execute the learning
        for epoch in range(epochs):
            sess.run(env_res_op)

            for i in range(steps_per_epoch):
                _, episode_finished, r = sess.run([one_step, done, reward])
                rewards[epoch] += r
                env.render_if_activated()

                # finish episode if necessary
                if episode_finished:
                    print("Episode {} finished with reward of {}".format(epoch, rewards[epoch]))
                    break

# now print the reward graph
fig = plt.figure("Rewards")
plt.plot(rewards)
plt.show()