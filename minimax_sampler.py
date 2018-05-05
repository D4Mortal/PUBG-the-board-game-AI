# Date                  : 01/05/2018
# Python version        : 3.6.4

# This is a test program that records the time to do minimax search with
# different number of pieces on the board

###############################################################################

from partb import Player

from random import randint
import time

SIZE = 8  # board size

UNOCC = 0  #'-'
WHITE = 1  #'O'
BLACK = 2  #'@'
CORNER = 3  #'X'
WALL = 4

black = 12
white = 12

result = {}
result2 = {}
def removeRandomPiece(state):
    global black
    global white
    removed = False
    while not removed:
        for row, line in enumerate(state):
            for col, symbol in enumerate(line):
                if symbol == WHITE or symbol == BLACK:
                    if randint(0,24) == 10:
                        if state[row, col] == WHITE:
                            white -= 1
                        else:
                            black -= 1
                        state[row, col] = UNOCC
                        removed = True
                        return


def removePieceInOrder(state, colour):
    for row, line in enumerate(state):
        for col, symbol in enumerate(line):
            if symbol == colour:
                state[row, col] = UNOCC
                return


# random specifies if the peices are removed randomly or in order

def testRun(random = False):

    game = Player('white')
    if random:
        file = open("test_result_random.txt", "a")

    else:
        file = open("test_result_in_order.txt", "a")
#    print('before update')
    game.put_piece(4, 3, WHITE)  # example for move
    game.put_piece(4, 7, WHITE)  # example for move
    game.put_piece(2, 5, WHITE)  # example for move
    game.put_piece(4, 6, WHITE)  # example for move
    game.put_piece(1, 1, WHITE)  # example for move
    game.put_piece(0, 3, WHITE)  # example for move
    game.put_piece(2, 0, WHITE)  # example for move
    game.put_piece(6, 3, WHITE)  # example for move
    game.put_piece(0, 5, WHITE)  # example for move
    game.put_piece(5, 0, WHITE)  # example for move
    game.put_piece(4, 1, WHITE)  # example for move
    game.put_piece(6, 7, WHITE)  # example for move


    game.put_piece(2, 4, BLACK)  # example for move
    game.put_piece(2, 2, BLACK)  # example for move
    game.put_piece(3, 5, BLACK)  # example for move
    game.put_piece(3, 6, BLACK)  # example for move
    game.put_piece(3, 1, BLACK)  # example for move
    game.put_piece(3, 3, BLACK)  # example for move
    game.put_piece(5, 4, BLACK)  # example for move
    game.put_piece(3, 5, BLACK)  # example for move
    game.put_piece(0, 1, BLACK)  # example for move
    game.put_piece(2, 7, BLACK)  # example for move
    game.put_piece(7, 1, BLACK)  # example for move
    game.put_piece(7, 4, BLACK)  # example for move
    game.put_piece(6, 6, BLACK)  # example for move

    depth = 1
    removed = 1

    while True:
        global black
        global white
        start = time.time()

        child_nodes_friendly = game.node.genChild(game.player_colour)
        child_nodes_enemy = game.node.genChild(game.opp_colour)
        total_branch = len(child_nodes_friendly) + len(child_nodes_enemy)

        game.miniMax(depth,child_nodes_friendly)
        end = time.time()
        timeTaken = end - start
#        file.write("#Black: {}, #White: {}, Depth: {}, Elapsed: {} seconds\n".format(black, white, depth, timeTaken))
        file.write("#Black braches: {}, #White branches: {}, total: {}, Depth: {}, Elapsed: {} seconds\n"
                   .format(len(child_nodes_friendly), len(child_nodes_enemy), total_branch, depth, timeTaken))

        if timeTaken < 0.5:
            if total_branch not in result:
                result[total_branch] = depth,timeTaken

            elif result[total_branch][0] < depth:
                result[total_branch] = depth,timeTaken
            elif result[total_branch][0] == depth:
                average = result[total_branch][1] + timeTaken
                average = average/2
                result[total_branch] = depth,average



        if (total_branch,depth) not in result:
            result2[total_branch,depth] = timeTaken
        elif result[(total_branch,depth)] == depth:
            average2 = result2[(total_branch,depth)] + timeTaken
            average2 = average2/2
            result2[(total_branch,depth)] = average


        if timeTaken > 2:
            depth = 1

            if not random:
                if removed % 2 == 1:
                    removePieceInOrder(game.node.state, BLACK)
                    black -= 1
                else:
                    removePieceInOrder(game.node.state, WHITE)
                    white -= 1
            else:
                removeRandomPiece(game.node.state)
            removed += 1
            continue
        else:
            depth += 1

        if black == 1 or white == 1:
            break
    file.close()
    return



for a in range(10):
    black = 12
    white = 12
    testRun(True)

final_results = open("branching_results.txt", "w")
for key, value in sorted(result.items(), reverse = True):
    final_results.write("{}:{},".format(key,value[0]))

final_results2 = open("branching_results_detailed_average.txt", "w")
for key, value in sorted(result2.items(), reverse = True):
    final_results2.write("{}:{}\n".format(key,value))

final_results.close()
final_results2.close()
print(result)
