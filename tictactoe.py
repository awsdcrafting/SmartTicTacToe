import sys

import numpy as np


class TicTacToe:
    def __init__(self):
        self.game_board = np.zeros((3, 3), dtype=np.int16)

    def make_move(self, player_id, x, y):
        if not self.game_board[y][x] == 0:
            sys.exit("Error: Tried to make a move on an already occupied tile")
        self.game_board[y][x] = player_id

    def possibilities(self):
        empty = np.where(self.game_board == 0)
        first = empty[1].tolist()
        second = empty[0].tolist()
        return [x for x in zip(first, second)]

    def check_if_won(self):
        winner = None
        for player_id in [1, -1]:
            if self.row_win(player_id):
                winner = player_id
            elif self.col_win(player_id):
                winner = player_id
            elif self.diag_win(player_id):
                winner = player_id
        if np.all(self.game_board != 0) and winner is None:
            winner = 0
        return winner

    def row_win(self, player_id):
        return np.any(np.all(self.game_board == player_id, axis=1))

    def col_win(self, player_id):
        return np.any(np.all(self.game_board == player_id, axis=0))

    def diag_win(self, player_id):
        diag1 = np.array([self.game_board[0, 0], self.game_board[1, 1], self.game_board[2, 2]])
        diag2 = np.array([self.game_board[0, 2], self.game_board[1, 1], self.game_board[2, 0]])
        return np.all(diag1 == player_id) or np.all(diag2 == player_id)

    def print_game_board(self):
        print("GameBoard:")
        for row in self.game_board:
            print(row)
        print()
