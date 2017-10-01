# simply import the numpy package.
import numpy as np
import tensorflow as tf
from src.rl_agent.agent.BootstrappedReplayMemory import BootstrappedReplayMemory
from src.rl_agent.network.KHeadMultipleActionDeepQNetwork import KHeadMultipleActionDeepQNetwork
from src.rl_agent.policy.Policy import Policy

from src.policies_nn.BestPolicy import BestPolicy
from src.util.spaces.DiscreteSpace import DiscreteSpace


class BootstrappedDDQNAgent:
    """This class represents a BootstrappedDDQNAgent and can be used to together with
    BootstrappedReplayMemory"""

    def __init__(self, state_space, action_space, policy, replay_memory, batch_size, copy_offset, num_heads=10, debug=False):
        """Constructs a BootstrappedDDQNAgent.

        Args:
            state_space: Give the discrete state space
            action_space: Give the discrete action space
            policy: Give the policies_nn the agent should use
            replay_memory: The replay memory the agent should sample values from
            batch_size: The batch_size, e.g. the number of samples from the replay memory
            copy_offset: The number of learning steps until the target model gets copied into the online model
        """

        # check if both spaces are derived from the correct type
        assert isinstance(state_space, DiscreteSpace)
        assert isinstance(action_space, DiscreteSpace)
        assert isinstance(policy, Policy)
        assert isinstance(replay_memory, BootstrappedReplayMemory)

        # set the internal debug variable
        self.deb = debug
        self.K = num_heads
        self.batch_size = batch_size
        self.copy_offset = copy_offset
        self.iteration = 0

        # Save the epsilon for the greedy policies_nn.
        self.policy = policy
        self.state_space = state_space
        self.action_space = action_space
        self.replay_memory = replay_memory

        # init necessary objects
        structure = 2 * [20]
        structure2 = [20, 10, 10]
        self.dqn = KHeadMultipleActionDeepQNetwork(state_space, structure, structure2, action_space, num_heads, 1.0, True)
        self.target_dqn = KHeadMultipleActionDeepQNetwork(state_space, structure, structure2, action_space, num_heads, 1.0, True)

        self.bp = BestPolicy(action_space.get_size())

        # create new session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def reset(self):
        """This method resets the replay memory."""

        # save the current position
        self.replay_memory.reset()

    # this method exploits the internal model
    def choose_action(self, cs, k, target=False):

        # apply defined policies_nn from the outside
        target_model = self.target_dqn if target else self.dqn
        q = target_model.eval(cs)
        return self.policy.select_action(q)

    # this method exploits the internal model
    def choose_best_action(self, cs, k, target=False):

        # sample the best action from the best policies_nn
        target_model = self.target_dqn if target else self.dqn
        q = target_model.eval(cs)
        return self.bp.select_action(q)

    # This method can be used to learn a tuple inside. Therefore
    # you just have to supply one tuple of training example.
    #
    # - discount: Here you can define the discount factor
    # - current_state: The current state of the environment
    # - reward: The reward received for this step.
    # - action: The action you took.
    # - next_state: The state you reached.
    # - sample_size: Here you define the sample size to take from the replay memory.
    #
    def learn_tuple(self, discount, current_state, reward, action, next_state):

        # one has to create the mask appropriately
        mask = np.zeros(self.K, dtype=np.float32)
        mask[np.random.randint(0, self.K, 3)] = 1

        # first convert to binary and then insert it in the
        # replay memory itself. Afterwards sample so many values
        # as specified by the sample_size parameter of the function.
        self.replay_memory.insert(current_state, reward, action, next_state, mask)
        [actions, rewards, c_states, n_states, masks] = self.replay_memory.sample(self.batch_size)

        # retrieve the correct length
        m = np.sum(masks)

        # retrieve the list of indices
        positions = np.where(masks == 1)
        ks = np.empty(m, dtype=np.int32)

        # create combined fields
        combined_actions = np.empty(m, dtype=np.int32)
        combined_c_states = np.empty(m, dtype=np.int32)
        combined_actionvalues = np.empty(m, dtype=np.float32)

        # iterate over all positions, which can
        # be mapped to the different heads
        c = 0
        for x, y in zip(positions[0], positions[1]):

            # use the learning network to sample the action
            t_action, _ = self.choose_best_action([n_states[y]], x, False)

            # get the q function for the
            q = self.target_dqn.eval([n_states[y]])
            filtered_q = q[x, t_action[0], 0]

            # Calculate the target values for this reinforcement learning method
            combined_actions[c] = actions[y]
            combined_c_states[c] = c_states[y]
            combined_actionvalues[c] = rewards[y] + discount * filtered_q
            ks[c] = x
            c = c + 1

        # if self.debug: print("state_action is \n" + str(np.hstack([c_state, action])))
        self.dqn.learn(combined_c_states, combined_actions, combined_actionvalues, ks)

        # the iteration handling
        self.iteration = self.iteration + 1
        if self.iteration == self.copy_offset:
            self.iteration = 0
            self.target_dqn.copy_weights(self.dqn)
