# Chirag Rao Sahib      : 836011
# Daniel Hao            : 834496
# Date                  :
# Python version        : 3.6.4

###############################################################################
import copy
from collections import defaultdict
from Board import board
import numpy as np


SIZE = 8  # board size

BLACK = 2  #'@'
WHITE = 1  #'O'
UNOCC = 0  #'-'
CORNER = -1  #'X'


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
        self.state = np.full((SIZE,SIZE), UNOCC, dtype=int)
        state[0,0] = CORNER
        state[7,7] = CORNER
        state[0,7] = CORNER
        state[7,0] = CORNER

        # [['X','-','-','-','-','-','-','X'],
        # ['-','-','-','-','-','-','-','-'],
        # ['-','-','-','-','-','-','-','-'],
        # ['-','-','-','-','-','-','-','-'],
        # ['-','-','-','-','-','-','-','-'],
        # ['-','-','-','-','-','-','-','-'],
        # ['-','-','-','-','-','-','-','-'],
        # ['X','-','-','-','-','-','-','X']]


###############################################################################


def testrun():
    game = Player(WHITE)
    print(game.state)

###############################################################################

testrun()

