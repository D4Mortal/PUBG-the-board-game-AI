import copy
import numpy as np
import sys
from collections import defaultdict


SIZE = 8  # board size


CORNER = -1  #'X'
UNOCC = 0  #'-'
WHITE = 1  #'O'
BLACK = 2  #'@'
WALL = 5

MAP = {1:2, 2:1}

MODS = {'R': (0, 1),  # how each direction modifies a position
        '2R': (0, 2),
        'L': (0, -1),
        '2L': (0, -2),
        'D': (1, 0),
        '2D': (2, 0),
        'U': (-1, 0),
        '2U': (-2, 0)}

###############################################################################
def posCheck(state, row, col, dir, return_rowcol =False):
        '''
        returns symbol at a given board position (modified by direction)
        '''
        if return_rowcol:
            return row + MODS[dir][0], col + MODS[dir][1]

        return state[row + MODS[dir][0], col + MODS[dir][1]]


###############################################################################

class Player():

    def __init__(self, colour):
        self.state = np.full((SIZE, SIZE), UNOCC, dtype=int)
        self.state[0,0] = CORNER
        self.state[0,7] = CORNER
        self.state[7,0] = CORNER
        self.state[7,7] = CORNER
        self.turns = 126
        self.totalTurns = 25

        if colour[0] == 'W':
          self.player_colour = WHITE
          self.node = board(self.state, None, WHITE)
          self.opp_colour = BLACK

        else:
          self.player_colour = BLACK
          self.opp_colour = WHITE
          self.node = board(self.state, None, BLACK)



    def put_piece(self, row, col, piece):
      self.state[row, col] = piece


    def action(self, turns):
        # turns since start of current phase
        # placing action = (x,y)
        # moving action = ((a,b),(c,d)) from a,b to c,d
        # forfeit action = None
        self.turns = turns + 1
        if self.totalTurns > 24:
            if self.countPieces(self.node) < 8:
                action = self.miniMax(5)
                self.node.makeMove(action, self.player_colour)
            else:
                action = self.miniMax(4)
                self.node.makeMove(action, self.player_colour)
            return action
        else:
            # placing phase 
            self.totalTurns += 1
            return


    def update(self, action):
        if self.node.state[action[0][0]][action[0][1]] <= 0:
            return None
        self.node.makeMove(action, self.opp_colour)
        self.totalTurns += 1


    def countPieces(self, node):
        unique, counts = np.unique(node.state, return_counts=True)
        results = dict(zip(unique, counts))
        return results[self.player_colour] + results[self.opp_colour]


    def initStrat(self):
      return
      # placeholder initialise strategy

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
        
    def miniMax(self, depth):

        def maxValue(node, depth, alpha, beta, turns):
            if turns == 129:
                self.firstShrink(node)
            if turns == 193:
                self.secondShrink(node)
                
            if node.isComplete():
                return node.calculateScore()
            if depth <= 0:
                return node.calculateScore()

            v = -np.inf

            for nextMoves in node.genChild(node.colour):
                
                v = max(v, minValue(nextMoves, depth-1, alpha, beta, turns+1))
#                print(nextMoves.calculateScore(), end='')
#                print("White's move: ", end='')
#                print(nextMoves.move)
#                print(nextMoves.state)
                if v >= beta:
                    return v
                alpha = max(alpha, v)

            return v


        def minValue(node, depth, alpha, beta, turns):
            if turns == 129:
                self.firstShrink(node)
            if turns == 193:
                self.secondShrink(node)
                
            if node.isComplete():
                return node.calculateScore()
            if depth <= 0:
                return node.calculateScore()

            v = np.inf

            for nextMoves in node.genChild(MAP[node.colour]):
                
                v = min(v, maxValue(nextMoves, depth-1, alpha, beta, turns+1))
#                print(nextMoves.calculateScore(), end='')


#                print(nextMoves.state)
                if v <= alpha:
                    return v
                beta = min(beta, v)

            return v

        best_score = -np.inf
        beta = np.inf
        best_action = None
        
        for Moves in self.node.genChild(self.player_colour):
            v = minValue(Moves, depth-1, best_score, beta, self.turns)
            if v > best_score:
                best_score = v
                best_action = Moves.move
        return best_action
###############################################################################

# simple board class that stores the current board config and the move that brought
# it there
# class inherits from object and uses slots instead of dict to reduce memory usuage
# and faster attribute access

class board(object):

    __slots__ = ('state', 'move', 'colour')

    def __init__(self, state, move, colour):
        self.state = state
        self.move = move
        self.colour = colour

###############################################################################

    # function that returns a new board object created from the specified move
    def newMakeMove(self, action):
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

        self.eliminateBoard(newState, colour)
        newBoard = board(newState, action, self.colour)
        return newBoard

###############################################################################

    def put_piece(self, state, row, col, piece):
      state[row, col] = piece

###############################################################################

    def isEliminated(self, board, row, col, piece):
        '''
        check whether the given piece will be eliminated by the corner
            and/or surrounding opponents
        '''
        if piece == WHITE:
            flag = BLACK
        if piece == BLACK:
            flag = WHITE

        if row == 0 or row == 7:
            checkLeft = posCheck(board, row, col, 'L')
            checkRight = posCheck(board, row, col, 'R')
            if checkLeft == flag or checkLeft == CORNER:
                if checkRight == flag or checkRight == CORNER:
                    return True

        elif col == 0 or col == 7:
            checkUp = posCheck(board, row, col, 'U')
            checkDown = posCheck(board, row, col, 'D')
            if checkUp == flag or checkUp == CORNER:
                if checkDown == flag or checkDown == CORNER:
                    return True

        else:
            # generate positions to check
            check = [posCheck(board,row,col,i) for i in ['L','R','U','D']]
            if check[0] == flag or check[0] == CORNER:
                if check[1] == flag or check[1] == CORNER:
                    return True
            if check[2] == flag or check[2] == CORNER:
                if check[3] == flag or check[3] == CORNER:
                    return True

        return False

###############################################################################

    def eliminateBoard(self, state, colour):
        '''
        returns updated board after necessary eliminations
        '''
        mapping = {1: [BLACK, WHITE], 2: [WHITE, BLACK]}
        for piece in mapping[colour]:
            for row, line in enumerate(state):
                for col, symbol in enumerate(line):
                    if symbol == piece:
                        if self.isEliminated(state, row, col, piece):
                            state[row][col] = UNOCC

        return state
###############################################################################

    # function that make moves on the current object, changes the current state,
    # does not create a new board

    def makeMove(self, action, colour):
        action_tuple = np.array(action)
        action_size = action_tuple.size


        if action_size == 1:
          return

        elif action_size == 2:
          #placing phase
          self.state[action_tuple[0], action_tuple[1]] = colour

        elif action_size == 4:
          # moving phase
          colour = self.state[action[0][0]][action[0][1]]
          self.put_piece(self.state, action_tuple[0][0], action_tuple[0][1], UNOCC)
          self.put_piece(self.state, action_tuple[1][0], action_tuple[1][1], colour)

        self.eliminateBoard(self.state, colour)
        self.move = action # update the move that brought it to this state
        return

###############################################################################

    def calculateScore(self):
        unique, counts = np.unique(self.state, return_counts=True)
        results = dict(zip(unique, counts))


        if self.colour in results and MAP[self.colour] in results:
            if results[self.colour] <= 2 and results[MAP[self.colour]] > 2:
                return -999
            if results[self.colour] > 2 and results[MAP[self.colour]] <= 2:
                return 999
            else:
                feature1 = results[self.colour] - results[MAP[self.colour]]
                feature2 = self.safeMobility(self.state)
                eval_func = 0.8 * feature1 + 0.5 * feature2

                return eval_func

    def isComplete(self):
        score = self.calculateScore()
        if score == 999 or score == -999:
            return True
        return False
###############################################################################

    def checkSurr(self, board, row, col, piece):
        availMoves = 0
        checkCond = {'D':[row+1 < SIZE, row+2 < SIZE],
                     'U':[row-1 >= 0, row-2 >= 0],
                     'R':[col+1 < SIZE, col+2 < SIZE],
                     'L':[col-1 >= 0, col-2 >= 0]}

        for m in checkCond:
            if checkCond[m][0]:

                row2, col2 = posCheck(board,row,col, m, return_rowcol=True)
                newPos = self.state[row,col]
                if newPos == UNOCC and self.isEliminated(board, row2, col2, piece) == False:
                    availMoves += 1
                if newPos == WHITE or newPos == BLACK:
                    if checkCond[m][1]:
                        row3, col3 = posCheck(board,row,col, '2' + m, return_rowcol=True)
                        if self.state[row3,col3] == UNOCC and self.isEliminated(board, row3, col3, piece)== False:
                            availMoves += 1
        return availMoves


    def safeMobility(self, board):
        playerMoves = 0
        oppMoves = 0
        row = 0

        for line in board:
            col = 0
            for symbol in line:
                if symbol == self.colour:
                    playerMoves += self.checkSurr(board, row , col, self.colour)
                if symbol == MAP[self.colour]:
                    oppMoves += self.checkSurr(board, row , col, MAP[self.colour])
                col += 1
            row += 1

        return playerMoves - oppMoves

###############################################################################
    def genChild(self, colour):

        row = 0
        action = []
        action_tuple = ()
        for r in self.state:
            col = 0
            for element in r:
                if element == colour:

                    if row + 1 < 8:
                        if posCheck(self.state, row, col, 'D') == UNOCC:

                            action_tuple = ((row, col), (row + 1, col))
                            action.append(self.newMakeMove(action_tuple))

                        elif posCheck(self.state, row, col, 'D') == WHITE or posCheck(self.state, row, col, 'D') == BLACK:
                            if row + 2 < 8:
                                if posCheck(self.state, row, col, '2D') == UNOCC:

                                     action_tuple = ((row, col), (row + 2, col))
                                     action.append(self.newMakeMove(action_tuple))

                    if row - 1 >= 0:
                        if posCheck(self.state, row, col, 'U') == UNOCC:

                            action_tuple = ((row, col), (row - 1, col))
                            action.append(self.newMakeMove(action_tuple))

                        elif posCheck(self.state, row, col, 'U') == WHITE or posCheck(self.state, row, col, 'U') == BLACK:
                            if row - 2 >= 0:
                                if posCheck(self.state, row, col, '2U') == UNOCC:

                                    action_tuple = ((row, col), (row - 2, col))
                                    action.append(self.newMakeMove(action_tuple))


                    if col + 1 < 8:
                        if posCheck(self.state, row, col, 'R') == UNOCC:

                            action_tuple = ((row, col), (row, col + 1))
                            action.append(self.newMakeMove(action_tuple))

                        elif posCheck(self.state, row, col, 'R') == WHITE or posCheck(self.state, row, col, 'R') == BLACK:
                            if col + 2 < 8:
                                if posCheck(self.state, row, col, '2R') == UNOCC:

                                    action_tuple = ((row, col), (row, col + 2))
                                    action.append(self.newMakeMove(action_tuple))

                    if col - 1 >= 0:
                        if posCheck(self.state, row, col, 'L') == UNOCC:

                            action_tuple = ((row, col), (row, col - 1))
                            action.append(self.newMakeMove(action_tuple))

                        elif posCheck(self.state, row, col, 'L') == WHITE or posCheck(self.state, row, col, 'L') == BLACK:
                            if col - 2 >= 0:
                                if posCheck(self.state, row, col, '2L') == UNOCC:

                                    action_tuple = ((row, col), (row, col - 2))
                                    action.append(self.newMakeMove(action_tuple))
                col += 1
            row += 1

        return action

###############################################################################
def testMemUsage():
    gameState = np.full((SIZE, SIZE), UNOCC, dtype=int)

#    print("show", sys.getsizeof(board(gameState, ((0,0),(0,1)))))
#    print(gameState )
#    print("show", sys.getsizeof(gameState[0]))

    l = [board(gameState, 'bar', WHITE) for i in range(50000000)]
    print(sys.getsizeof(l))

###############################################################################

def testrun(me = 'WHITE'):
    game = Player(me)

    # update board tests
    move = ((0,1), (3,4))
    move2 = ((3,4), (6,6))
    move3 = ((3,5), (4,5))
    move4 = ((6,6),(5,6))
    place = (6,5)
    null_move = None

#    print('before update')
    game.put_piece(4, 3, WHITE)  # example for move
    game.put_piece(2, 4, BLACK)  # example for move
    game.put_piece(2, 2, BLACK)  # example for move
    game.put_piece(4, 7, WHITE)  # example for move
    game.put_piece(2, 5, WHITE)  # example for move
    game.put_piece(4, 6, WHITE)  # example for move
    game.put_piece(3, 5, BLACK)  # example for move
    game.put_piece(3, 6, BLACK)  # example for move
#    print(game.node.state)
#    print(game.player_colour)
#    print(game.node.calculateScore())
#
#    print(game.node.isComplete())

#    for a in game.node.genChild(BLACK):
#        print("this this generated")
#        print(a.state)
#        print(a.calculateScore())


#    print("This is the current board config")
#    print(game.node.state)
#    depth = input("Please select a depth to search on: ")
#    print("Searching ahead for {} moves...".format(depth))
#    result = game.miniMax(int(depth))
#    print("The optimal move for white is: ", end='')
#    print(result)
    
#    print("this is the current board state at turn 126")
#    print(game.node.state)
#    game.action(126)
#    print("The ideal move would be: {} for turn 127".format(game.node.move))

    
#    game.firstShrink()
#    print(game.node.state)
#    print(game.node.calculateScore())
#    
#    game.secondShrink()
#    print(game.node.state)
#    print(game.node.calculateScore())
    
#    game.update(((2, 5), (4, 5)))
#    print(game.node.state)
#    print(game.node.calculateScore())
    
    
    
testrun()
#testMemUsage()
