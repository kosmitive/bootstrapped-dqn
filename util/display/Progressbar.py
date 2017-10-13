class Progressbar:

    """A simple progressbar"""
    def __init__(self, m, s):

        # print s times =
        print(s * "=")
        self.m = m
        self.current = 0
        self.dist = m / s
        self.next = self.dist

    # this moves the progressbar one element further.
    def progress(self, num):

        self.current = self.current + num

        # if we reached the maximum already
        if self.current == self.m: return

        # as long as the current is longer than the next
        while self.current >= self.next:
            print('=', end='', flush=True)
            self.next = self.next + self.dist





