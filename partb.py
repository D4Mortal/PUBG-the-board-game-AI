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

###############################################################################

class Player():

    def __init__(self, colour):
        self.state = np.full((SIZE, SIZE), UNOCC, dtype=int)
        self.state[0,0] = CORNER
        self.state[0,7] = CORNER
        self.state[7,0] = CORNER
        self.state[7,7] = CORNER
        
        self.node = board(self.state, None)
        
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
        self.node.makeMove(action, self.opp_colour)
          

    def initStrat(self):
      return
      # placeholder initialise strategy


    def miniMax(self):
        # uses board class as a node to generate 
        return
    
    
    def heuristics(self):
        return


###############################################################################
        
# simple board class that stores the current board config and the move that brought
# it there
# class inherits from object and uses slots instead of dict to reduce memory usuage
# and faster attribute access
        
class board(object):
 
    __slots__ = ('state', 'move', 'score')
 
    def __init__(self, state, move):
        self.state = state
        self.move = move
        self.score = self.calculateScore()
 
###############################################################################
        
     # function that returns the new board object created from the specified move
    def newMakeMove(self, action, colour):
        newState = copy.deepcopy(self.state)
 
        action_tuple = np.array(action)
        action_size = action_tuple.size

        if action_size == 1:
          return

        elif action_size == 2:
          #placing phase
          newState[action_tuple[0], action_tuple[1]] = self.opp_colour

        elif action_size == 4:
          # moving phase
          self.put_piece(newState, action_tuple[0][0], action_tuple[0][1], UNOCC)
          self.put_piece(newState, action_tuple[1][0], action_tuple[1][1], colour)
 
        
        newBoard = board(newState, action)
        return newBoard
 
###############################################################################
        
    def put_piece(self, state, row, col, piece):
      state[row, col] = piece
      

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
          self.state[action_tuple[0], action_tuple[1]] = self.opp_colour

        elif action_size == 4:
          # moving phase
          self.put_piece(self.state, action_tuple[0][0], action_tuple[0][1], UNOCC)
          self.put_piece(self.state, action_tuple[1][0], action_tuple[1][1], colour)
 
        
        self.move = action # update the move that brought it to this state
        return
    
###############################################################################
        
    def calculateScore(self):
        return 4
    
###############################################################################

def testMemUsage():
    gameState = np.full((SIZE, SIZE), UNOCC, dtype=int)

#    print("show", sys.getsizeof(board(gameState, ((0,0),(0,1)))))
#    print(gameState )
#    print("show", sys.getsizeof(gameState[0]))

    l = [board(gameState, 'bar') for i in range(50000000)]
    print(sys.getsizeof(l))
    
###############################################################################
    
def testrun(me = 'WHITE'):
    game = Player(me)

    # update board tests
    move = ((0,1), (3,4))
    move2 = ((3,4), (6,6))
    place = (6,5)
    null_move = None

    print('before update')
    game.put_piece(0, 1, BLACK)  # example for move
    print(game.node.state)

    print('after update')
    game.update(move)
    print(game.node.state)
    
    print('after update 2')
    game.update(move2)
    print(game.node.state)

testrun()
testMemUsage()