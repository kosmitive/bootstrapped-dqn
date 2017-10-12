class ContinuousSpace:
    def __init__(self, D, IB, IE):
        """Simply the boundaries and the dimension"""
        assert D == len(IB) and D == len(IE)
        self.D = D
        self.intervals = list(zip(IB, IE))

    def dim(self):
        return self.D