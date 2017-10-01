class Environment:

    def __init__(self, name, N):
        """Creates a new Environment which keeps track of N states."""
        self.env_name = name
        self.N = N

    def observation_space(self):
        """Only the space as an object."""
        raise NotImplementedError()

    def action_space(self):
        raise NotImplementedError()

    def current_observation_graph(self):
        """Can be integrated into tensorflow."""
        raise NotImplementedError()

    def step_graph(self, action):
        """Passes back the next reward and state."""
        raise NotImplementedError()

    def apply_step_graph(self):
        """This applies the next state saves it internally"""
        raise NotImplementedError()

    def random_walk(self, steps):
        """This executes a random walk for x steps"""
        raise NotImplementedError()

    def render(self, D):
        """This method can be used to render an environment"""
        print("For this problem was no rendering mechanism supplied.")

    def reset_graph(self):
        """This graph resets the internal state etc."""
        raise NotImplementedError()

