# Date                  : 01/05/2018
# Python version        : 3.6.4

# This is a sampling program that records the time to do minimax search with
# different number of pieces on the board and branching factor

###############################################################################

from partb import Player, board
import numpy as np
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
###############################################################################

def removeRandomPiece(state):
    global black
    global white
    removed = False
    while not removed:
        for row, line in enumerate(state):
            for col, symbol in enumerate(line):
                if symbol == WHITE or symbol == BLACK:
                    if randint(0,63) == 10:
                        if state[row, col] == WHITE:
                            white -= 1
                        else:
                            black -= 1
                        state[row, col] = UNOCC
                        removed = True
                        return

###############################################################################
                        
def removePieceInOrder(state, colour):
    for row, line in enumerate(state):
        for col, symbol in enumerate(line):
            if symbol == colour:
                state[row, col] = UNOCC
                return
            
###############################################################################
                                    
def generateRandomBoard(whiteNum, blackNum):
    state = np.full((SIZE, SIZE), UNOCC, dtype=int)
    state[0,0] = CORNER
    state[0,7] = CORNER
    state[7,0] = CORNER
    state[7,7] = CORNER
    temp = board(state,(),WHITE)
    
    while not (whiteNum == 0 and blackNum == 0):
        for row, line in enumerate(state):
            for col, symbol in enumerate(line):
                if symbol == UNOCC:
                    random = randint(0,127)
                    if random == 1 or random == 2:
                        if random % 2 == 0 and whiteNum > 0:
                            state[row, col] = WHITE
                            isElim = temp.eliminate_board(state, WHITE)
                            if len(isElim) == 0:
                                whiteNum -= 1
                            else:
                                whiteNum -= 1
                                for killed in isElim:
                                    if killed[1] == WHITE:
                                        whiteNum += 1
                                    else:
                                        blackNum += 1
                       
                        elif random % 2 != 0 and blackNum > 0:
                            state[row, col] = BLACK
                            isElim = temp.eliminate_board(state, BLACK)
                            if len(isElim) == 0:
                                blackNum -= 1
                            else:
                                blackNum -= 1
                                for killed in isElim:
                                    if killed[1] == WHITE:
                                        whiteNum += 1
                                    else:
                                       blackNum += 1
    return state
###############################################################################
    
# random specifies if the peices are removed randomly or in order
def sampling(random = False):

    game = Player('white')
    if random:
        file = open("test_result_random.txt", "a")

    else:
        file = open("test_result_in_order.txt", "a")

    game.node.state = generateRandomBoard(12,12)
    print(game.node.state)
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

###############################################################################

for numOfRuns in range(1):
    black = 12
    white = 12
    sampling(True)


with open("branching_results.txt", "w") as final_results:
    for key, value in sorted(result.items(), reverse = True):
        final_results.write("{}:{},".format(key,value[0]))

with open("branching_results_detailed_average.txt", "w") as final_results2:
    for key, value in sorted(result2.items(), reverse = True):
        final_results2.write("{}:{}\n".format(key,value))
    


print(result)
