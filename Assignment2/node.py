class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.score = 0
        self.traversals = 0

    def get_probability(self):
        if self.traversals == 0:
            return 0
        else:
            return self.score / self.traversals

    def update(self, delta_score):
        self.score += delta_score
        self.traversals += 1

    @property
    def has_children(self):
        return bool(self.children)

    def __str__(self):
        return f'Node: ({self.score}/{self.traversals}), {self.state}'


