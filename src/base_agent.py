"""
Each agent play as  player N: 1
So any time agent is making a move, you need use correct env
- if needed you may need to reverse the env.board
"""
class BaseAgent:
    def choose_action(self, env):
        raise NotImplementedError