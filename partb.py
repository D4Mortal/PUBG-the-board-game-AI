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
        
        
        
        if colour[0] == 'w':
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
 
    __slots__ = ('state', 'move', 'score', 'colour')
 
    def __init__(self, state, move, colour):
        self.state = state
        self.move = move
        self.colour = colour
        self.score = self.calculateScore()
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
          newState[action_tuple[0], action_tuple[1]] = self.opp_colour

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
 
        self.eliminateBoard(self.state, colour)
        self.move = action # update the move that brought it to this state
        return
    
###############################################################################
        
    def calculateScore(self):
        blackCount = 0
        whiteCount = 0
        for row in self.state:
            for symbol in row:
                if symbol == BLACK:
                    blackCount += 1
                if symbol == WHITE:
                    whiteCount += 1
        if self.colour == WHITE: return whiteCount - blackCount
        else: return blackCount - whiteCount
###############################################################################    
    
    def isComplete(self):
        unique, counts = np.unique(self.state, return_counts=True)
        results = dict(zip(unique, counts))
        if results[WHITE] <= 2 or results[BLACK] <= 2:
            return True
        return False
    
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
    move3 = ((3,5), (4,5))
    move4 = ((6,6),(5,6))
    place = (6,5)
    null_move = None

    print('before update')
    game.put_piece(0, 1, BLACK)  # example for move
    print()
    game.put_piece(4, 7, WHITE)  # example for move
    game.put_piece(4, 4, WHITE)  # example for move
    game.put_piece(4, 6, WHITE)  # example for move
    game.put_piece(3, 5, BLACK)  # example for move
    game.put_piece(3, 6, BLACK)  # example for move
    print(game.node.state)

    print('after update')
    game.update(move)
    print(game.node.state)
    
    print('after update 2')
    game.update(move2)
    print(game.node.state)
    
    print('after update 3')
    game.update(move3)
    print(game.node.state)
    
    print('after update 4')
    game.update(move4)
    print(game.node.state)
    
    print(game.node.isComplete())
    
#    for a in game.node.genChild(BLACK):
#        print("this this generated")
#        print(a.state)
    
testrun()
#testMemUsage()
selfColour = WHITE
oppColour = BLACK