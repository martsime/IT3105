from copy import deepcopy

class NimState:
    def __init__(self, number_of_stones, remove_stones_max, next_player=1):
        self.number_of_stones = number_of_stones
        self.remove_stones_max = remove_stones_max
        self.next_player = next_player

    def get_actions(self):
        return range(min(self.remove_stones_max, self.number_of_stones), 0, -1)

    def do_action(self, action):
        self.number_of_stones -= action
        if self.next_player == 1:
            self.next_player = 2
        else:
            self.next_player = 1

    @property
    def game_over(self):
        return self.number_of_stones == 0

    @property
    def winner(self):
        if self.game_over:
            if self.next_player == 2:
                return 1
            else:
                return 2

    def __str__(self):
        return f'Stones={self.number_of_stones}, NextPlayer={self.next_player}'
    
    def verbose(self, next_player, best_action):
        print(f'Player {next_player} takes {best_action} stones', end=' ')
        print(f'==> {self.number_of_stones} stones left.', end=' ')

    def __repr__(self):
        return f'{self.number_of_stones},{self.next_player}'

    def get_copy(self):
        return NimState(self.number_of_stones, self.remove_stones_max, self.next_player)


class TicTacToe:
    def __init__(self, board, next_player=1):
        self.board = board
        self.next_player = next_player


    def get_actions(self):
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    actions.append(i*3 + j)

        return actions

    def player_char(self):
        if self.next_player == 1:
            return 'X'
        else:
            return 'O'

    def do_action(self, action):
        i = action // 3
        j = action % 3
        self.board[i][j] = self.player_char()

        if self.next_player == 1:
            self.next_player = 2
        else:
            self.next_player = 1

    
    @property
    def game_over(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return True
        if self.board[2][0] == self.board[1][1] == self.board[0][2] != ' ':
            return True

        if not self.get_actions():
            return True
        return False

    @property
    def winner(self):
        if self.game_over:
            for i in range(3):
                if self.board[i][0] == self.board[i][1] == self.board[i][2] == 'X':
                    return 1
                if self.board[i][0] == self.board[i][1] == self.board[i][2] == 'O':
                    return 2
                if self.board[0][i] == self.board[1][i] == self.board[2][i] != 'X':
                    return 1
                if self.board[0][i] == self.board[1][i] == self.board[2][i] != 'O':
                    return 2
            if self.board[0][0] == self.board[1][1] == self.board[2][2] != 'X':
                return 1
            if self.board[2][0] == self.board[1][1] == self.board[0][2] != 'X':
                return 1
            if self.board[0][0] == self.board[1][1] == self.board[2][2] != 'O':
                return 2
            if self.board[2][0] == self.board[1][1] == self.board[0][2] != 'O':
                return 2
            
            return 'tie'


    def __str__(self):
        s = ''
        for row in self.board:
            s += '-'*7 + '\n'
            s += '|' + '|'.join(row) + '|\n'
        s += '-'*7 + '\n'
        return s
    
    def verbose(self, next_player, best_action):
        print(self)

    def __repr__(self):
        s = ''
        for row in self.board:
            s += ''.join(row)
        s += str(self.next_player)
        return s

    def get_copy(self):
        board = deepcopy(self.board)
        return TicTacToe(board, self.next_player)

