from copy import deepcopy
import random

import time
from settings import Settings
from state import Hex
from nodemanager import NodeManager
from simulation import Simulation
from policy import Policy
import queue
from threading import Thread
from app import Application


class Game:
    def __init__(self, game_settings=None):
        self.game_settings = game_settings

        self.state = None
        self.node_managers = {}

        self.tree_policy = None
        self.score_policy = None
        self.stats = {}
        self.app = None

        self.init_policies()
        self.setup()

    def setup(self):
        self.node_managers[1] = NodeManager()
        self.node_managers[2] = NodeManager()
        if self.game_settings['game'] == 'hex':
            self.setup_hex()
        if self.app:
            self.app.root.quit()
        self.app = Application(start_state=self.state)

    def start(self):
        for i in range(1, self.game_settings.get('G', 10) + 1):
            self.play(i)

    def setup_hex(self):
        if self.game_settings['P'] == 'random':
            next_player = random.randint(1, 2)
        else:
            next_player = self.game_settings['P']

        size = self.game_settings['size']
        board = Hex.initial_board(size)
        self.state = Hex(board, next_player=next_player)

    def init_policies(self):
        self.tree_policy = Policy.Tree.get(self.game_settings.get('tree_policy'))
        self.score_policy = Policy.Score.get(self.game_settings.get('score_policy'))

    def display_stats(self):
        total_games = self.game_settings.get('G')
        for key in self.stats.keys():
            wins = self.stats.get(key)
            percentage = wins / total_games * 100
            print(f'Player {key} won {wins} of {total_games} games ({percentage:.2f}%)')

    def play(self, number):
        verbose = self.game_settings.get('verbose')
        if verbose:
            print(f'GAME NUMBER {number}:')

            print(f'Start state: {self.state}')
        while not self.state.game_over:
            next_player = self.state.next_player
            node_manager = self.node_managers.get(next_player)
            best_action = self.simulate_best_action(self.state)
            self.state.do_action(best_action)
            self.app.update(self.state)
            if verbose:
                new_node = node_manager.get_node(self.state)
                self.state.verbose(next_player, best_action)
                print(f'Action stats: ({new_node.score}/{new_node.traversals}) = {new_node.get_probability() * 100:.2f}%')

        winner = self.state.winner
        print(f'Player {winner} won game {number}!\n')
        self.stats[winner] = self.stats.get(winner, 0) + 1

        self.setup()

    def simulate_best_action(self, state):
        node_manager = self.node_managers[self.state.next_player]
        simulation = Simulation(
            start_state=deepcopy(state),
            node_manager=node_manager,
            tree_policy=self.tree_policy,
            score_policy=self.score_policy,
            iterations=self.game_settings.get('M')
        )
        simulation.run()
        root_node = node_manager.get_node(self.state)
        best_action = Policy.Tree.best(root_node)
        return best_action


def main():
    game_settings = Settings.hex()
    game = Game(game_settings=game_settings)
    game.start()
    game.display_stats()


if __name__ == "__main__":
    main()
