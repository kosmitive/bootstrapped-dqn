class Environment:

    def observation_space(self):
        raise NotImplementedError()

    def action_space(self):
        raise NotImplementedError()

    def current_observation_graph(self):
        raise NotImplementedError()

    def step_graph(self, action):
        raise NotImplementedError()

    def apply_step_graph(self):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

    def reset_graph(self):
        raise NotImplementedError()

