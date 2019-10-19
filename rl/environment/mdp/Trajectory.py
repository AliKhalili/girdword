from collections import namedtuple


class Trajectory:
    def __init__(self, **kwargs):
        super(Trajectory, self).__init__()
        self._trajectory = {}
        self._record = namedtuple('trajectory', ['time_step', 'reward', 'state', 'action', 'next_state'])

    def add_trajectory(self, time_step, reward, state, action, next_state):
        self._trajectory[time_step] = self._record(time_step, reward, state, action, next_state)

    def reset(self):
        self._trajectory = {}

    def get_all(self):
        return self._trajectory

    def get(self, time_step):
        return self._trajectory[time_step]

    def length(self):
        return len(self._trajectory)

    def reward(self):
        return sum([rec.reward for rec in self._trajectory.values()])

    def history(self):
        return self.length(), self.reward(), self.get_all()
