from rl.prediction.PredictionBase import PredictionBase
import rl.common.Constant as CNSTNT


class IterativeDP(PredictionBase):
    def __init__(self, env, discount_factor, step_size):
        super().__init__(env, discount_factor, step_size)
        self.trajectory = {}

    def run(self, number_of_run):
        for episode in range(number_of_run):
            for state in range(self.state_space):
                if self.env.is_terminal(state):
                    continue
                v = 0
                for action in range(self.action_space):
                    self.env.state(state)
                    state_next, reward, is_terminal, info = self.env.step(CNSTNT.ACTIONS_VALUES[action])
                    action_prop = self.PI[state][action]
                    v += action_prop * (reward + self.discount_factor * self.V[state_next])
                self.V[state] = v
            self.trajectory[episode] = self.V.copy()
        super()._save()
        return self.trajectory
