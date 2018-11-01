
directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))


class Hex:
    def __init__(self, board, turn=1, next_player=1):
        self.turn = turn
        self.board = board
        self.size = len(board)
        self.next_player = next_player

    @staticmethod
    def initial_board(size=5):
        return [[0 for _ in range(size)] for _ in range(size)]
    
    def get_actions(self):
        actions = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    actions.append((i, j))
        return actions
    
    def do_action(self, action):
        x, y = action
        self.board[x][y] = self.next_player
        self.turn += 1
        if self.next_player == 1:
            self.next_player = 2
        else:
            self.next_player = 1

    def get_copy(self):
        new_board = [[x for x in row] for row in self.board]
        return Hex(board=new_board, turn=self.turn, next_player=self.next_player)
    
    def __repr__(self):
        s = ''
        for row in self.board:
            s += ''.join([str(x) for x in row])
        s += str(self.next_player)
        return s

    @property
    def winner(self):
        if self.turn < self.size * 2:
            return False
        start_nodes = []
        for i, row in enumerate(self.board):
            for j, cell in enumerate(row):
                if cell != 0 and (i == 0 or j == 0):
                    start_nodes.append((i, j, cell))

        for start_node in start_nodes:
            queue = [start_node]
            visited = [(start_node[0], start_node[1])]
            while len(queue):
                node = queue.pop()
                x, y, cell = node
                for direction in directions:
                    new_x = x + direction[0]
                    new_y = y + direction[1]
                    if new_x < 0 or new_y < 0 or new_x >= self.size or new_y >= self.size:
                        continue
                    if (new_x, new_y) in visited:
                        continue
                    if self.board[new_x][new_y] == start_node[2]:
                        if (start_node[0] == 0 and new_x == self.size - 1) or (start_node[1] == 0 and new_y == self.size - 1):
                            return cell
                        new_node = (new_x, new_y, cell)
                        visited.append((new_x, new_y))
                        queue.append(new_node)

        return False
    
    @property
    def game_over(self):
        if self.winner:
            return True
        else:
            return False

    def verbose(self, next_player, best_action):
        print(self)

