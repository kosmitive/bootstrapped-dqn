class DeepQNetwork:

    def copy_graph(self, network):
        raise NotImplementedError()

    def eval_graph(self, states):
        raise NotImplementedError()

    def learn_graph(self, learning_rate, Q, target_actions):
        raise NotImplementedError()

    def grid_graph(self, grid_dims, D):
        raise NotImplementedError()