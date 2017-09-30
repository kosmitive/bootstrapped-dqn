class Memory:
    def reset_graph(self):
        raise NotImplementedError()

    def store_and_sample_graph(self, current_state, next_state, action, reward, done):
        raise NotImplementedError()

    def store_graph(self, current_state, next_state, action, reward, done):
        raise NotImplementedError()