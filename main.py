import random
import sys

from ai import TicTacToePlayer
from neural_network import NeuralNetwork, crossover
from tictactoe import TicTacToe

team_size = 100
brain_structure = [9, 12, 12, 1]
iterations = 1000


def main(args):
    print("Program started with the following args: {}\n".format(args))

    red_pool = []  # ID:   1
    blue_pool = []  # ID: - 1

    # Creating initial population
    for current_network in range(0, team_size):
        red_pool.append(NeuralNetwork(brain_structure))
        blue_pool.append(NeuralNetwork(brain_structure))

    for current_iteration in range(0, iterations):
        red_gene_pool = []
        blue_gene_pool = []

        total_red_score = 0
        total_blue_score = 0

        games_won_red = 0
        games_won_blue = 0

        games_lost_red = 0
        games_lost_blue = 0

        draw_red = 0
        draw_blue = 0

        # Getting the score for all red networks
        for red_brain in red_pool:
            for blue_brain in blue_pool:
                result = get_winner(red_brain, blue_brain)
                if result == 1:  # The red won

                    red_brain.score += 2
                    blue_brain.score -= 2

                    total_red_score += 2
                    total_blue_score -= 2

                    games_won_red += 1
                    games_lost_blue += 1

                elif result == -2:  # Draw

                    draw_red += 1
                    draw_blue += 1

                else:  # The blue won

                    red_brain.score -= 2
                    blue_brain.score += 2

                    total_red_score -= 2
                    total_blue_score += 2

                    games_lost_red += 1
                    games_won_blue += 1

        for red_brain in red_pool:
            red_gene_pool += ([red_brain] * max(1, red_brain.score))
        for blue_brain in blue_pool:
            blue_gene_pool += ([blue_brain] * max(1, blue_brain.score))

        # Clearing the old red pool
        red_pool.clear()
        blue_pool.clear()

        # Creating the pools
        for current_new_network_id in range(0, team_size):
            red_pool.append(crossover(random.choice(red_gene_pool), random.choice(red_gene_pool)))
            # blue_pool.append(crossover(random.choice(blue_gene_pool), random.choice(blue_gene_pool)))
            blue_pool.append(NeuralNetwork(brain_structure))

        print("Generation {}".format(current_iteration))
        print("Score red:\t{} \t({}/{}/{}) ({})"
              .format(total_red_score, games_won_red, draw_red, games_lost_red, len(red_gene_pool)))
        print("Score blue:\t{} \t({}/{}/{}) ({})"
              .format(total_blue_score, games_won_blue, draw_blue, games_lost_blue, len(blue_gene_pool)))
        print()


def get_winner(red_brain, blue_brain):
    ai_0 = TicTacToePlayer(red_brain, 1)
    ai_1 = TicTacToePlayer(blue_brain, -1)

    game = TicTacToe()

    counter = 0

    if random.randint(1, 100) <= 50:
        counter += 1

    while game.check_if_won() == 0:
        if counter % 2 == 0:  # Player 1 Turn
            ai_0.make_move(game)
        else:
            ai_1.make_move(game)
        counter += 1

    return game.check_if_won()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])

    # neural_network_1 = NeuralNetwork([9, 11, 11, 4])
    # neural_network_2 = NeuralNetwork([9, 11, 11, 4])

    # new_network = crossover(neural_network_1, neural_network_2)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
