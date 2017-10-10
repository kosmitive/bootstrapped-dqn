class Policy:
    """Represents a basic policies_nn. It therefore needs a function, which
    evaluates successfully the q function"""


    def choose_action(self, q):
        """This method of a policies_nn basically gets a Q function
        and has to return the action to take now. Here you can
        specify behaviour like taking the best action, or sometimes
        different actions to them.

        Args:
            q: The q function to use for evaluating.

        Returns: The index of the action that should be taken
        """
        raise NotImplementedError("Please implement choose_action method")