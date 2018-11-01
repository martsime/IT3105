import random


class Simulation:
    def __init__(self, start_state, node_manager, tree_policy, score_policy, iterations=1000):
        self.start_state = start_state
        self.node_manager = node_manager
        self.tree_policy = tree_policy
        self.score_policy = score_policy
        self.iterations = iterations
        self.visited = []

    def run(self):
        for i in range(1, self.iterations + 1):
            self.search()

    def search(self):
        self.visited = []
        current_node = self.node_manager.get_node(self.start_state)
        self.visited.append(current_node)
        while current_node.has_children:
            action = self.tree_policy(current_node)
            next_node = current_node.children[action]
            self.visited.append(next_node)
            current_node = next_node

        if current_node.state.game_over:
            self.backprop(current_node.state.winner)
        else:
            self.node_manager.expand_node(current_node)
            action = self.tree_policy(current_node)
            next_node = current_node.children[action]
            self.visited.append(next_node)
            self.rollout(next_node)

    def rollout(self, current_node):
        state = current_node.state.get_copy()
        while not state.game_over:
            action = random.choice(state.get_actions())
            state.do_action(action)

        self.backprop(winner=state.winner)

    def backprop(self, winner):
        while self.visited:
            current_node = self.visited.pop()
            did_player_win = winner != current_node.state.next_player
            current_node.update(delta_score=self.score_policy(win=did_player_win))



