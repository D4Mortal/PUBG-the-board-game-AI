import copy
import numpy as np
import sys

from collections import defaultdict

import time
import random
import timeit



SIZE = 8  # board size

UNOCC = 0  #'-'
WHITE = 1  #'O'
BLACK = 2  #'@'
CORNER = 3  #'X'
WALL = 4

MINIMAX_DEPTH = 2

PHASE1 = 24
PHASE2 = PHASE1 + 128
PHASE3 = PHASE2 + 64

MAP = {WHITE:BLACK, BLACK:WHITE}



MODS = {'R': (0, 1),  # how each direction modifies a position
        '2R': (0, 2),
        'L': (0, -1),
        '2L': (0, -2),
        'D': (1, 0),
        '2D': (2, 0),
        'U': (-1, 0),
        '2U': (-2, 0),
        'N' : (0,0)}

###############################################################################

def pos_check(state, row, col, dir, return_rowcol = False):
    '''
    returns symbol at a given board position (modified by direction)
    '''
    x, y = row + MODS[dir][0], col + MODS[dir][1]

    if return_rowcol:
        return x, y

    return state[x, y]


###############################################################################

def initTable():
     ZobristTable = np.empty((SIZE, SIZE, 5))
     for i in range(SIZE):
         for j in range(SIZE):
             for k in range(5):
                 ZobristTable[i,j,k] = random.randint(0,1e19)

    # print(np.random.choice(10000, size = (SIZE, SIZE, 5), replace=False))
     return ZobristTable



###############################################################################

# Zobrist Hashing
def zorHash(state, table):
    value = 0
    for i in range(SIZE):
        for j in range(SIZE):
            if state[i, j] == WHITE or state[i, j] == BLACK:
                piece = state[i, j]
                value = value^int(table[i, j, piece])


    return value

###############################################################################

def hashMove(hashValue, state, action):
    originPos = action[0]
    targetPos = action[1]
    newHash = copy.deepcopy(hashValue)
    newHash = newHash^int(ZOR[originPos[0], originPos[1], state[originPos[0]][originPos[1]]])
    newHash = newHash^int(ZOR[targetPos[0], targetPos[1], state[originPos[0]][originPos[1]]])
    return newHash

###############################################################################

def hashRemove(hashValue, state, position):
    newHash = copy.deepcopy(hashValue)
    newHash = newHash^int(ZOR[position[0], position[1], state[position[0]][position[1]]])
    return newHash

###############################################################################

class Player():

    def __init__(self, colour):
        self.state = np.full((SIZE, SIZE), UNOCC, dtype=int)
        self.state[0,0] = CORNER
        self.state[0,7] = CORNER
        self.state[7,0] = CORNER
        self.state[7,7] = CORNER
        self.turns = 0
        self.totalTurns = 0#PHASE1 + 1

        if colour[0] == 'w':
          self.player_colour = WHITE
          self.opp_colour = BLACK
          self.node = board(self.state, None, WHITE)
          self.place_moves = [(2,0),(2,7),(4,0),(4,7),(5,0),(5,7),(0,2),(0,5)]

        else:
          self.player_colour = BLACK
          self.opp_colour = WHITE
          self.node = board(self.state, None, BLACK)
          self.place_moves = [(5,0),(5,7),(3,0),(3,7),(2,0),(2,7),(7,2),(7,5)]

###############################################################################

    def put_piece(self, row, col, piece):
      self.state[row, col] = piece

###############################################################################

    def action(self, turns):
        # This is only used by player pieces
        self.turns = turns + 1
        if self.totalTurns > PHASE1:
            if self.countPieces(self.node) < 8:
                action = self.miniMax(MINIMAX_DEPTH)
                self.node.update_board_inplace(action, self.player_colour)
            else:
                action = self.miniMax(MINIMAX_DEPTH)
                self.node.update_board_inplace(action, self.player_colour)
            return action
        else:
           # print('activate placing')
            self.totalTurns += 1
            place_move = self.place_phase()
            self.node.update_board_inplace(place_move, self.player_colour)

            return place_move

###############################################################################
    def in_danger(self, piece):
        # return where to place to block danger
        # checkCond = {'D':[row+1 < SIZE, row+2 < SIZE],
        #              'U':[row-1 >= 0, row-2 >= 0],
        #              'R':[col+1 < SIZE, col+2 < SIZE],
        #              'L':[col-1 >= 0, col-2 >= 0]}

        checkCond = dict()

        for row, line in enumerate(self.state):
            checkCond['D'] = row+1 < SIZE
            checkCond['U'] = row-1 >= 0
            for col, symbol in enumerate(line):
                checkCond['R'] = col+1 < SIZE
                checkCond['L'] = col-1 >= 0
                if symbol == piece:
                    for m in checkCond:
                        if checkCond[m]:
                            row2, col2 = pos_check(board,row,col, m, return_rowcol=True)
                            newPos = self.state[row2,col2]
                            if newPos == MAP[piece]:
                                if row2 == row:
                                    if col2 > col:
                                        if self.state[row, col-1] == UNOCC:
                                            return (row, col-1)
                                    if col2 < col:
                                        if self.state[row, col+1] == UNOCC:
                                            return (row, col+1)
                                    # row opposite
                                if col2 == col:
                                    if row2 > row:
                                        if self.state[row-1, col] == UNOCC:
                                            return (row-1, col)
                                    if row2 < row:
                                        if self.state[row+1, col] == UNOCC:
                                            return (row+1, col)
        return None



    def place_phase(self):

        if self.state[self.place_moves[0][0], self.place_moves[0][1]] == UNOCC:
            return self.place_moves[0]
        if self.state[self.place_moves[1][0], self.place_moves[1][1]] == UNOCC:
            return self.place_moves[1]

# (self, board, row, col, piece)
        while self.totalTurns < 21:
            if self.state[self.place_moves[2][0], self.place_moves[2][1]] == UNOCC and not self.node.is_eliminated(self.state, self.place_moves[2][0], self.place_moves[2][1], self.player_colour):
                return self.place_moves[2]
            if self.state[self.place_moves[3][0], self.place_moves[3][1]] == UNOCC and not self.node.is_eliminated(self.state, self.place_moves[3][0], self.place_moves[3][1], self.player_colour):
                return self.place_moves[3]
            if self.state[self.place_moves[2][0], self.place_moves[2][1]] == self.player_colour and self.state[self.place_moves[4][0], self.place_moves[4][1]] == UNOCC:
                return self.place_moves[4]
            if self.state[self.place_moves[3][0], self.place_moves[3][1]] == self.player_colour and self.state[self.place_moves[5][0], self.place_moves[5][1]] == UNOCC:
                return self.place_moves[5]

            danger_result = self.in_danger(self.player_colour)
            if danger_result != None:
                return danger_result

            kill_result = self.in_danger(self.opp_colour)
            if kill_result != None:
                return kill_result

            if danger_result == None and kill_result == None: break


        if self.state[self.place_moves[6][0], self.place_moves[6][1]] == UNOCC:
            return self.place_moves[6]

        if self.state[self.place_moves[7][0], self.place_moves[7][1]] == UNOCC:
            return self.place_moves[7]


    # This is only called by enemy pieces
    def update(self, action):

        self.node.update_board_inplace(action, self.opp_colour)
        self.totalTurns += 1

###############################################################################

    def countPieces(self, node):
        results = np.bincount(node.state.ravel())
        return results[WHITE] + results[BLACK]

###############################################################################

    def firstShrink(self, node):
        node.state[0, :] = WALL
        node.state[7, :] = WALL
        node.state[:, 0] = WALL
        node.state[:, 7] = WALL
        node.state[1,1] = CORNER
        node.state[1,6] = CORNER
        node.state[6,1] = CORNER
        node.state[6,6] = CORNER

    def secondShrink(self, node):
        node.state[1, :] = WALL
        node.state[7, :] = WALL
        node.state[:, 1] = WALL
        node.state[:, 6] = WALL
        node.state[2,2] = CORNER
        node.state[2,5] = CORNER
        node.state[5,2] = CORNER
        node.state[5,5] = CORNER

###############################################################################

    def miniMax(self, depth):
        start = time.time()

        def maxValue(node, depth, alpha, beta, turns):
            if turns == 129:
                self.firstShrink(node)
            if turns == 193:
                self.secondShrink(node)

            if  depth <= 0 or node.is_terminal():
                return node.eval_node()

            v = -np.inf
            ordered_child_nodes = sorted(node.genChild(node.colour),
                key=lambda x: x.move_estim, reverse=True)

            for child in ordered_child_nodes:

                v = max(v, minValue(child, depth-1, alpha, beta, turns+1))
#                print(child.eval_node(), end='')
#                print("White's move: ", end='')
#                print(child.move)
#                print(child.state)
                if v >= beta:
                    return v
                alpha = max(alpha, v)

            return v


        def minValue(node, depth, alpha, beta, turns):
            if turns == 129:
                self.firstShrink(node)
            if turns == 193:
                self.secondShrink(node)

            if node.is_terminal():
                return node.eval_node()
            if depth <= 0:
                return node.eval_node()

            v = np.inf

            ordered_child_nodes = sorted(node.genChild(MAP[node.colour]),
                key=lambda x: x.move_estim)

            for child in ordered_child_nodes:

                v = min(v, maxValue(child, depth-1, alpha, beta, turns+1))
#                print(child.eval_node(), end='')
#                print(child.state)
                if v <= alpha:
                    return v
                beta = min(beta, v)

            return v

        best_score = -np.inf
        beta = np.inf
        best_action = None

        for child in self.node.genChild(self.player_colour):
            v = minValue(child, depth-1, best_score, beta, self.turns)
            if v > best_score:
                best_score = v
                best_action = child.move

        end = time.time()
        print(end - start)
        return best_action
###############################################################################

# simple board class that stores the current board config and the move that brought
# it there
# class inherits from object and uses slots instead of dict to reduce memory usuage
# and faster attribute access

class board(object):

    __slots__ = ('state', 'move', 'colour', 'move_estim')

    def __init__(self, state, move, colour):
        self.state = state
        self.move = move
        self.colour = colour
        self.move_estim = self.pvs_estim()
###############################################################################

    # function that returns a new board object created from the specified move
    def update_board_return(self, action):
        newState = copy.deepcopy(self.state)
        action_tuple = np.array(action)
        action_size = action_tuple.size

        if action_size == 1:
          return

        elif action_size == 2:
          #placing phase
          newState[action_tuple[0], action_tuple[1]] = self.opp_colour ############################## need to fix this

        elif action_size == 4:
          # moving phase
          colour = self.state[action[0][0]][action[0][1]]
          self.put_piece(newState, action_tuple[0][0], action_tuple[0][1], UNOCC)
          self.put_piece(newState, action_tuple[1][0], action_tuple[1][1], colour)

        self.eliminate_board(newState, colour)

        return board(newState, action, self.colour)

###############################################################################

    def put_piece(self, state, row, col, piece):
      state[row, col] = piece

###############################################################################

    def is_eliminated(self, board, row, col, piece):
        '''
        check whether the given piece will be eliminated by the corner
            and/or surrounding opponents
        '''
        if piece == WHITE:
            flag = BLACK
        if piece == BLACK:
            flag = WHITE

        if row == 0 or row == 7:
            checkLeft = pos_check(board, row, col, 'L')
            checkRight = pos_check(board, row, col, 'R')
            if checkLeft == flag or checkLeft == CORNER:
                if checkRight == flag or checkRight == CORNER:
                    return True

        elif col == 0 or col == 7:
            checkUp = pos_check(board, row, col, 'U')
            checkDown = pos_check(board, row, col, 'D')
            if checkUp == flag or checkUp == CORNER:
                if checkDown == flag or checkDown == CORNER:
                    return True

        else:
            # generate positions to check
            check = [pos_check(board,row,col,i) for i in ['L','R','U','D']]
            if check[0] == flag or check[0] == CORNER:
                if check[1] == flag or check[1] == CORNER:
                    return True
            if check[2] == flag or check[2] == CORNER:
                if check[3] == flag or check[3] == CORNER:
                    return True

        return False

###############################################################################



    def eliminate_board(self, state, colour):
        '''
        returns updated board after necessary eliminations
        '''

        # numpy ufunc ???
        mapping = {WHITE: [BLACK, WHITE], BLACK: [WHITE, BLACK]}
        # order of elimination

        for piece in mapping[colour]:
            for row, line in enumerate(state):
                for col, symbol in enumerate(line):
                    if symbol == piece:
                        if self.is_eliminated(state, row, col, piece):
                            state[row][col] = UNOCC

        return state

###############################################################################

    # function that make moves on the current object, changes the current state,
    # does not create a new board

    def update_board_inplace(self, action, colour):
        action_tuple = np.array(action)
        action_size = action_tuple.size


        if action_size == 1:
          return

        elif action_size == 2:
          #placing phase
          self.state[action_tuple[0], action_tuple[1]] = colour

        elif action_size == 4:
          # moving phase
          colour = self.state[action[0][0], action[0][1]]
          self.put_piece(self.state, action_tuple[0][0], action_tuple[0][1], UNOCC)
          self.put_piece(self.state, action_tuple[1][0], action_tuple[1][1], colour)

        self.eliminate_board(self.state, colour)
        self.move = action # update the move that brought it to this state
        return

###############################################################################

    def eval_node(self):
        results = np.bincount(self.state.ravel())
        if results[self.colour] <= 2 and results[MAP[self.colour]] > 2:
            return -999
        if results[self.colour] > 2 and results[MAP[self.colour]] <= 2:
            return 999
        else:
            f1= results[self.colour] - results[MAP[self.colour]]
            f2 = self.safeMobility(self.state)
            eval_func = 0.8 * f1 + 0.5 * f2

            return eval_func

###############################################################################

    def pvs_estim(self):
        '''
        principal variation estimation function
        '''
        results = np.bincount(self.state.ravel())
        return results[self.colour] - results[MAP[self.colour]]

###############################################################################

    def is_terminal(self):
        score = self.eval_node()
        if score == 999 or score == -999:
            return True
        return False

###############################################################################

    def count_legal_moves(self, board, row, col, piece):
        # ****
        availMoves = 0
        checkCond = {'D':[row+1 < SIZE, row+2 < SIZE],
                     'U':[row-1 >= 0, row-2 >= 0],
                     'R':[col+1 < SIZE, col+2 < SIZE],
                     'L':[col-1 >= 0, col-2 >= 0]}

        for m in checkCond:
            if checkCond[m][0]:
                row2, col2 = pos_check(board, row, col, m, return_rowcol=True)
                newPos = self.state[row2,col2]

                if newPos == UNOCC and not self.is_eliminated(board, row2, col2, piece):
                    availMoves += 1
                if newPos == WHITE or newPos == BLACK:
                    if checkCond[m][1]:
                        row3, col3 = pos_check(board,row,col, '2' + m, return_rowcol=True)
                        if self.state[row3,col3] == UNOCC and not self.is_eliminated(board, row3, col3, piece):
                            availMoves += 1
        return availMoves

###############################################################################

    def safeMobility(self, board):
        playerMoves = 0
        oppMoves = 0
        oppColour = MAP[self.colour]

        for row, line in enumerate(board):
            for col, symbol in enumerate(line):
                if symbol == self.colour:
                    playerMoves += self.count_legal_moves(board, row , col, self.colour)
                if symbol == oppColour:
                    oppMoves += self.count_legal_moves(board,row,col, oppColour)

        return playerMoves - oppMoves

###############################################################################

    def genChild(self, colour):
        child_nodes = []
        action_tuple = ()

        for row, line in enumerate(self.state):
            # describe up/down moves to check
            checkCond = {'D': [row+1 < SIZE, 1, 0, row+2 < SIZE, 2, 0],
                         'U': [row-1 >= 0, -1, 0, row-2 >= 0, -2, 0]}

            for col, symbol in enumerate(line):
                # describe left/right moves to check
                checkCond['R'] = [col+1 < SIZE, 0, 1, col+2 < SIZE, 0, 2]
                checkCond['L'] = [col-1 >= 0, 0, -1, col-2 >= 0, 0, -2]

                if symbol == colour:
                    for dir in checkCond:
                        if checkCond[dir][0]:
                            posToCheck = pos_check(self.state, row, col, dir)

                            if posToCheck == UNOCC:
                                tmpA = row + checkCond[dir][1]
                                tmpB = col + checkCond[dir][2]

                                action_tuple = ((row, col), (tmpA, tmpB))
                                child_nodes.append(self.update_board_return(action_tuple))

                            elif posToCheck == WHITE or posToCheck == BLACK:
                                # check whether jump is possible
                                if checkCond[dir][3]:
                                    j = '2' + dir  # jump direction
                                    if pos_check(self.state,row,col,j) == UNOCC:
                                        tmpA = row + checkCond[dir][4]
                                        tmpB = col + checkCond[dir][5]

                                        action_tuple = ((row, col), (tmpA, tmpB))
                                        child_nodes.append(self.update_board_return(action_tuple))

        return child_nodes

###############################################################################
def testMemUsage():
    gameState = np.full((SIZE, SIZE), UNOCC, dtype=int)

#    print("show", sys.getsizeof(board(gameState, ((0,0),(0,1)))))
#    print(gameState )
#    print("show", sys.getsizeof(gameState[0]))

    l = [board(gameState, 'bar', WHITE) for i in range(50000000)]
    print(sys.getsizeof(l))

###############################################################################

def testrun(me = 'white'):
    game = Player(me)

    # update board tests
    move = ((0,1), (3,4))
    move2 = ((3,4), (6,6))
    move3 = ((3,5), (1,1))
    move4 = ((6,6),(5,6))
    place = (6,5)
    null_move = None

#    print('before update')
    # game.put_piece(4, 3, WHITE)  # example for move
    # game.put_piece(2, 4, BLACK)  # example for move
    # game.put_piece(2, 2, BLACK)  # example for move
    # game.put_piece(4, 7, WHITE)  # example for move
    # game.put_piece(2, 5, WHITE)  # example for move
    # game.put_piece(4, 6, WHITE)  # example for move
    # game.put_piece(3, 5, BLACK)  # example for move
    # game.put_piece(3, 6, BLACK)  # example for move
#    print(game.node.state)
#    print(game.player_colour)
#    print(game.node.eval_node())
#
#    print(game.node.is_terminal())

#    for a in game.node.genChild(BLACK):
#        print("this this generated")
#        print(a.state)
#        print(a.eval_node())


    # print("This is the current board config")
    # print(game.node.state)
    # depth = input("Please select a depth to search on: ")
    # print("Searching ahead for {} moves...".format(depth))
    # result = game.miniMax(int(depth))
    # print("The optimal move for white is: ", end='')
    # print(result)
#
    # print("this is the current board state")
    # print(game.node.state)

    print('place test')

    for i in list(range(0,24,2)):
        print('game move', i)
        if i == 12:
            game.put_piece(2, 4, BLACK)
            game.put_piece(2,5, WHITE)
        print('total_turns', game.totalTurns)
        print(game.node.state)
        game.action(i)
  #  print("The ideal move would be: {} for turn 127".format(game.node.move))


#    game.firstShrink()
#    print(game.node.state)
#    print(game.node.eval_node())
#
#    game.secondShrink()
#    print(game.node.state)
#    print(game.node.eval_node())

#    game.update(((2, 5), (4, 5)))
#    print(game.node.state)

#    print(game.node.calculateScore())


    # print(game.node.eval_node())


    r = zorHash(game.node.state, ZOR)
    print(r)


    print(hashMove(r, game.node.state, ((3,5), (1,1))))


    game.update(move3)              # move3 is ((3,5), (1,1))
    a = zorHash(game.node.state, ZOR)
    print(a)

    game.node.state[4,6] = UNOCC
    a = zorHash(game.node.state, ZOR)
    print(a)
    print(r^int(ZOR[4, 6, UNOCC]))



if __name__ == "__main__":
    print (timeit.timeit('"zorHash(state,table)".join(str(n) for n in range(100))',number=100))
    print (timeit.timeit('"hash(state,table)".join(str(n) for n in range(100))',number=100))

ZOR = initTable()
testrun()
#testMemUsage()
