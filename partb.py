import copy
import numpy as np
import sys
from collections import defaultdict

SIZE = 8  # board size


CORNER = -1  #'X'
UNOCC = 0  #'-'
WHITE = 1  #'O'
BLACK = 2  #'@'


MAX_NODES = 1e5  # max number of nodes to expand
MAX_DEPTH = 5e2  # max depth of tree to explore
DEADEND = 1e5  # heavy (bad) score for deadends (heuristic)
BLACK_MULTIPLIER = 1e3  # heavy (bad) score for more black pieces (heuristic)


MODS = {'R': (0, 1),  # how each direction modifies a position
        '2R': (0, 2),
        'L': (0, -1),
        '2L': (0, -2),
        'D': (1, 0),
        '2D': (2, 0),
        'U': (-1, 0),
        '2U': (-2, 0)}



class Player():

    def __init__(self, colour):
        self.state = np.full((SIZE, SIZE), UNOCC, dtype=int)
        self.state[0,0] = CORNER
        self.state[0,7] = CORNER
        self.state[7,0] = CORNER
        self.state[7,7] = CORNER

        if colour[0] == 'w':
          self.player_colour = WHITE
          self.opp_colour = BLACK
        else:
          self.player_colour = BLACK
          self.opp_colour = WHITE




    def put_piece(self, row, col, piece):
      self.state[row, col] = piece

    def action(self, turns):
        # turns since start of current phase
        # placing action = (x,y)
        # moving action = ((a,b),(c,d)) from a,b to c,d
        # forfeit action = None
        return 0

    def update(self, action):
        action_tuple = np.array(action)
        action_size = action_tuple.size

        if action_size == 1:
          return

        elif action_size == 2:
          #placing phase
          self.state[action_tuple[0], action_tuple[1]] = self.opp_colour

        elif action_size == 4:
          # moving phase
          self.put_piece(action_tuple[0][0], action_tuple[0][1], UNOCC)
          self.put_piece(action_tuple[1][0], action_tuple[1][1], self.opp_colour)

    def initStrat(self):
      return
      # placeholder initialise strategy




# simple board class that stores the current board config and the move that brought
# it there
# class inherits from object and uses slots instead of dict to reduce memory usuage
# and faster attribute access

# class board(object):

#     __slots__ = ('state', 'move')

#     def __init__(self, state, move):
#         self.state = state
#         self.move = move



#     # function that returns the new board object created from the specified move
#     def newMakeMove(self, move):
#         newState = copy.deepcopy(self.state)

#         # make changes in the newState according to the moves specified

#         newBoard = board(newState, move)
#         return newBoard



#     # function that make moves on the current object, changes the current state,
#     # does not create a new board
#     def _makeMove(self, move):


#         # make changes in the self.state according to the moves specified


#         return




# gameState = []
# for i in range(8):
#     gameState.append(input().split())

# print("show", sys.getsizeof(board(gameState, ((0,0),(0,1)))))
# print(gameState )
# print("show", sys.getsizeof(gameState[0]))

# l = [board(gameState, 'bar') for i in range(20000000)]


def testrun(me = 'WHITE'):
    game = Player(me)

    # update board tests
    move = ((0,1), (3,4))
    place = (6,5)
    null_move = None

    print('before update')
    game.put_piece(0, 1, BLACK)  # example for move
    print(game.state)

    print('after update')
    game.update(move)
    print(game.state)

testrun()
