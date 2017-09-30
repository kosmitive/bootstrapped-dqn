class Agent:
    def register_memory(self, memory):
        raise NotImplementedError()

    def register_policy(self, policy):
        raise NotImplementedError()

    def action_graph(self, current_observation):
        raise NotImplementedError()

    def observe_graph(self, current_observation, next_observation, action, reward, done):
        raise NotImplementedError()