# Chirag Rao Sahib      : 836011
# Daniel Hao            : 834496
# Date                  : 
# Python version        : 3.6.4

###############################################################################
import copy
from collections import defaultdict

MAX_NODES = 1e5  # max number of nodes to expand
MAX_DEPTH = 5e2  # max depth of tree to explore
DEADEND = 1e5  # heavy (bad) score for deadends (heuristic)
BLACK_MULTIPLIER = 1e3  # heavy (bad) score for more black pieces (heuristic)
BLACK = '@'
WHITE = 'O'
UNOCC = '-'
CORNER = 'X'
SIZE = 8  # board size
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
        self.colour = colour
        self.initializeBoard()
        
###############################################################################
        
    def action(self, turns):
        return 0
    

###############################################################################
        
    def update(self, aciton):
        return 0

###############################################################################  
        
    def initializeBoard(self):
        self.state = [['X','-','-','-','-','-','-','X'],
                      ['-','-','-','-','-','-','-','-'],
                      ['-','-','-','-','-','-','-','-'],
                      ['-','-','-','-','-','-','-','-'],
                      ['-','-','-','-','-','-','-','-'],
                      ['-','-','-','-','-','-','-','-'],
                      ['-','-','-','-','-','-','-','-'],
                      ['X','-','-','-','-','-','-','X']]
        
        
############################################################################### 
        
        
def testrun():
    game = Player(WHITE)
    print(game.state)
    
###############################################################################     
    
testrun()
    