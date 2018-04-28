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
def posCheck(state, row, col, dir):
        '''
        returns symbol at a given board position (modified by direction)
        '''
        return state[row + MODS[dir][0], col + MODS[dir][1]]
    
    
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
        if self.node.state[action[0][0]][action[0][1]] <= 0:
            return None
        self.node.makeMove(action)
          

    def initStrat(self):
      return
      # placeholder initialise strategy


    def miniMax(self):
        # uses board class as a node to generate 
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
    def newMakeMove(self, action):
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
          colour = self.state[action[0][0]][action[0][1]]
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
    
    def makeMove(self, action):
        action_tuple = np.array(action)
        action_size = action_tuple.size
        
        
        if action_size == 1:
          return

        elif action_size == 2:
          #placing phase
          self.state[action_tuple[0], action_tuple[1]] = self.opp_colour

        elif action_size == 4:   
          # moving phase
          colour = self.state[action[0][0]][action[0][1]]
          self.put_piece(self.state, action_tuple[0][0], action_tuple[0][1], UNOCC)
          self.put_piece(self.state, action_tuple[1][0], action_tuple[1][1], colour)
 
        
        self.move = action # update the move that brought it to this state
        return
    
###############################################################################
        
    def calculateScore(self):
        return 4
###############################################################################    
    
    def isComplete(self):
        unique, counts = np.unique(self.state, return_counts=True)
        results = dict(zip(unique, counts))
        if results[WHITE] <= 2 or results[BLACK] <= 2:
            return True
        return False
    
###############################################################################
    def genChild(self):
        
        row = 0
        actions = defaultdict(list)
        action = []
        action_tuple = ()
        for r in self.state:
            col = 0
            for element in r:
                if element == WHITE:

                    if row + 1 < 8:
                        if posCheck(self.state, row, col, 'D') == UNOCC:
                            actions[str(row) + str(col)].append(str(row + 1) + str(col))
                            action_tuple = ((row, col), (row +1, col))
                            action.append(self.newMakeMove(action_tuple))
        
                        elif posCheck(self.state, row, col, 'D') == WHITE or posCheck(self.state, row, col, 'D') == BLACK:
                            if row + 2 < 8:
                                if posCheck(self.state, row, col, '2D') == UNOCC:
                                     actions[str(row) + str(col)].append(str(row + 2) + str(col))
                                     action_tuple = ((row, col), (row +1, col))
                                     action.append(self.newMakeMove(action_tuple))
                    if row - 1 >= 0:
                        if posCheck(self.state, row, col, 'U') == UNOCC:
                            actions[str(row) + str(col)].append(str(row - 1) + str(col))

        
                        elif posCheck(self.state, row, col, 'U') == WHITE or posCheck(self.state, row, col, 'U') == BLACK:
                            if row - 2 >= 0:
                                if posCheck(self.state, row, col, '2U') == UNOCC:
                                    actions[str(row) + str(col)].append(str(row - 2) + str(col))
            
                    
                    
                    if col + 1 < 8:
                        if posCheck(self.state, row, col, 'R') == UNOCC:
                            actions[str(row) + str(col)].append(str(row) + str(col + 1))
        
                        elif posCheck(self.state, row, col, 'R') == WHITE or posCheck(self.state, row, col, 'R') == BLACK:
                            if col + 2 < 8:
                                if posCheck(self.state, row, col, '2R') == UNOCC:
                                    actions[str(row) + str(col)].append(str(row) + str(col + 2))
                    
                    if col - 1 >= 0:
                        if posCheck(self.state, row, col, 'L') == UNOCC:
                            actions[str(row) + str(col)].append(str(row) + str(col - 1))
        
                        elif posCheck(self.state, row, col, 'L') == WHITE or posCheck(self.state, row, col, 'L') == BLACK:
                            if col - 2 >= 0:
                                if posCheck(self.state, row, col, '2L') == UNOCC:
                                    actions[str(row) + str(col)].append(str(row) + str(col - 2))
    
                col += 1
            row += 1
            
        return actions
        

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
    game.put_piece(4, 4, WHITE)  # example for move
    print(game.node.state)

    print('after update')
    game.update(move)
    print(game.node.state)
    
    print('after update 2')
    game.update(move2)
    print(game.node.state)
    
    print(game.node.isComplete())
    print(game.node.genChild())
testrun()
#testMemUsage()