import copy
import sys

import numpy as np


class TicTacToePlayer:

    def __init__(self, neural_network, player_id):
        self.neural_network = neural_network
        self.player_id = player_id

    def make_move(self, tic_tac_toe_game):

        possible_moves = tic_tac_toe_game.possibilities()
        # print("Possible moves: {}".format(possible_moves))

        best_move = None
        best_score = None

        for possible_move in possible_moves:
            game_board_after_move = copy.deepcopy(tic_tac_toe_game.game_board)
            game_board_after_move[possible_move[1]][possible_move[0]] = self.player_id

            network_input = np.reshape(game_board_after_move, [9, 1])

            network_output = self.neural_network.calculate(network_input)

            if best_score is None or network_output > best_score:
                best_move = possible_move
                best_score = network_output

        if best_move is None or best_score is None:
            sys.exit("Error: Could not find the best move")

        # print("Best move: {}".format(best_move))

        tic_tac_toe_game.make_move(self.player_id, best_move[0], best_move[1])
