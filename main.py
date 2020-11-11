import concurrent
import random
import sys
from concurrent.futures.process import ProcessPoolExecutor

from ai import TicTacToePlayer
from neural_network import NeuralNetwork, crossover
from tictactoe import TicTacToe
from concurrent import futures

team_size = 250
brain_structure = [9, 6, 6, 1]


# iterations = 10000000


def main(args):
    print("Program started with the following args: {}\n".format(args))

    red_pool = []  # ID:   1
    blue_pool = []  # ID: - 1

    # Creating initial population
    for current_network in range(0, team_size):
        red_pool.append(NeuralNetwork(brain_structure))
        blue_pool.append(NeuralNetwork(brain_structure))

    current_iteration = 0

    while True:
        red_gene_pool = []
        blue_gene_pool = []

        executor = ProcessPoolExecutor(12)
        my_future = [executor.submit(evaluate_brain, red_brain, blue_pool) for red_brain in red_pool]
        concurrent.futures.wait(my_future)

        best_score = None

        for b_id in range(0, len(red_pool)):
            score = my_future[b_id].__getattribute__("_result")
            if best_score is None or score > best_score:
                best_score = score
            red_gene_pool += ([red_pool[b_id]] * max(1, int(score)))

        # Clearing the old red pool
        red_pool.clear()
        # blue_pool.clear()

        # Creating the pools
        for current_new_network_id in range(0, team_size):
            red_pool.append(crossover(random.choice(red_gene_pool), random.choice(red_gene_pool)))
            # blue_pool.append(crossover(random.choice(blue_gene_pool), random.choice(blue_gene_pool)))
            # blue_pool.append(NeuralNetwork(brain_structure))

        current_iteration += 1

        print("Generation {}:\t{} ({})".format(current_iteration, best_score, len(red_gene_pool)))
        # print("Score red:\t\t{}/{}/{} ({})".format(games_won_red, draw_red, games_lost_red, len(red_gene_pool)))
        # print("Score blue:\t\t{}/{}/{} ({})".format(games_won_blue, draw_blue, games_lost_blue, len(blue_gene_pool)))


def evaluate_brain(red_brain, blue_pool):
    for blue_brain in blue_pool:
        result = get_winner(red_brain, blue_brain)
        if result == 1:  # The red won
            red_brain.score += 10.0
        elif result == 0:  # Draw
            red_brain.score += 5
        else:  # The blue won
            red_brain.score -= 10.0
    return red_brain.score


def get_winner(red_brain, blue_brain):
    ai_0 = TicTacToePlayer(red_brain, 1)
    ai_1 = TicTacToePlayer(blue_brain, -1)

    game = TicTacToe()

    counter = 0

    if random.randint(1, 100) <= 50:
        counter += 1

    while game.check_if_won() is None:
        if counter % 2 == 0:  # Player 1 Turn
            ai_0.make_move(game)
        else:
            ai_1.make_move(game)
        counter += 1

    return game.check_if_won()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])

    # neural_network_1 = NeuralNetwork([2, 3, 2], 0)
    # neural_network_2 = NeuralNetwork([2, 3, 2], 10)

    # neural_network_1.print_network()
    # neural_network_2.print_network()

    # new_network = crossover(neural_network_1, neural_network_2)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
