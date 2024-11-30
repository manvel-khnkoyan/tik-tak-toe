class BaseAgent:
    def choose_action(self, state):
        raise NotImplementedError

    def learn(self, state, action, reward, next_state):
        raise NotImplementedError