import numpy as np


class BootstrappedReplayMemory:
    """This class represent a basic replay memory, which can be used
    in the setting of reinforcement learning.
    """

    def __init__(self, N, K):
        """Basic constructor for replay memory

        Args:
            N: Represents the size of the replay memory.
        K: The number of heads used for that replay memory.
        """

        # init the corresponding vectors
        self.actions = np.empty(N, dtype=np.int32)
        self.rewards = np.empty(N, dtype=np.int32)
        self.cStates = np.empty(N, dtype=np.int32)
        self.nStates = np.empty(N, dtype=np.int32)
        self.bootstrap_masks = np.empty((K, N), dtype=np.int32)

        # save the current position
        self.N = N
        self.count = 0
        self.current = 0

    def reset(self):
        """This method resets the replay memory."""

        # save the current position
        self.count = 0
        self.current = 0

    def insert(self, current_state, reward, action, next_state, mask):
        """This method inserts a new tuple into the replay memory.

        Args:
            current_state: The current_state in a binary encoded fashion.
            reward: The reward for the action taken
            action: The action that was taken for the reward
            next_state: The state after the action was executed.
            mask: The mask for which it is used.
        """

        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.cStates[self.current] = current_state[0]
        self.nStates[self.current] = next_state[0]
        self.bootstrap_masks[:, self.current] = mask

        self.current = (self.current + 1) % self.N

        if self.count < self.N:
            self.count = self.count + 1

    def sample(self, A):
        """This method samples from the replay memory. Where one element
        could be sampled twice.

        Args:
            A: The number of samples to take from the replay memory.

        Returns:
            The obtained samples.
        """

        assert A <= self.N

        # get unique indices
        indices = np.random.choice(self.count, np.min([A, self.count]), replace=False)

        # pass back the sampled actions
        return self.actions[indices], self.rewards[indices], self.cStates[indices], self.nStates[indices], self.bootstrap_masks[:, indices]
