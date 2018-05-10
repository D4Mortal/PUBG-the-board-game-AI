import copy
import numpy as np
import random

SIZE = 8  # board size
UNOCC = 0  #'-'
WHITE = 1  #'O'
BLACK = 2  #'@'
CORNER = 3  #'X'
WALL = 4

PHASE1 = 23
PHASE2 = PHASE1 + 128
PHASE3 = PHASE2 + 64

WIN = 9999
LOSE = -1 * WIN
TIE = 1000

WEIGHTS = [1000, 5, 0.2, 2]

IDEAL_DEPTH = {80:2,79:2,78:2,77:2,76:2,75:2,74:2,73:2,72:2,71:2,70:2,69:2,68:2,
               67:2,66:2,65:2,64:2,63:2,62:2,61:2,60:2,59:2,58:2,57:2,56:2,55:2,
               54:2,53:3,52:3,51:3,50:3,49:2,48:2,47:2,46:3,45:3,44:3,43:3,42:3,
               41:3,40:3,39:3,38:3,37:3,36:3,35:3,34:3,33:3,32:3,31:3,30:3,29:3,
               28:3,27:3,26:4,25:4,24:4,23:4,22:4,21:4,20:4,19:5,18:5,17:5,16:4,
               15:4,14:5,13:5,12:5,11:5,10:5,9:6,8:6,7:6,6:6,5:7,4:7,3:7,2:7}

MAP = {WHITE:BLACK, BLACK:WHITE}

DEATHMAP= {WHITE: [6,7], BLACK: [0,1]}

PLACEMAP_WHITE = [[0,0,0,0,0,0,0,0],
                  [0,0,1,1,1,1,0,0],
                  [0,1,2,3,3,2,1,0],
                  [0,1,2,4,4,2,1,0],
                  [0,1,3,4,4,3,1,0],
                  [0,1,1,1,1,1,1,0],
                  [0,0,1,1,1,1,0,0],
                  [0,0,0,0,0,0,0,0]]

PLACEMAP_BLACK = [[0,0,0,0,0,0,0,0],
                  [0,0,1,1,1,1,0,0],
                  [0,1,1,1,1,1,1,0],
                  [0,1,2,4,4,2,1,0],
                  [0,1,2,4,4,2,1,0],
                  [0,1,2,2,2,2,1,0],
                  [0,0,1,1,1,1,0,0],
                  [0,0,0,0,0,0,0,0]]

PLACEMAP_WHITE2 =[[-1,-1,-1,-1,-1,-1,-1,-1],
                  [-1,-1,0,0,0,0,-1,-1],
                  [-1,0,1,2,2,1,0,-1],
                  [-1,0,2,4,4,2,0,-1],
                  [-1,0,3,4,4,3,0,-1],
                  [-1,0,0,1,1,0,0,-1],
                  [-1,-1,0,0,0,0,-1,-1],
                  [-1,-1,-1,-1,-1,-1,-1,-1]]

PLACEMAP_BLACK2 =[[-1,-1,-1,-1,-1,-1,-1,-1],
                  [-1,-1,0,0,0,0,-1,-1],
                  [-1,0,0,1,1,0,0,-1],
                  [-1,0,3,4,4,3,0,-1],
                  [-1,0,2,4,4,2,0,-1],
                  [-1,0,1,2,2,1,0,-1],
                  [-1,-1,0,0,0,0,-1,-1],
                  [-1,-1,-1,-1,-1,-1,-1,-1]]

CHECK_ORDER_WHITE = [(3,3),(4,3),(3,4),(4,4),(3,2),(3,5),(4,2),(4,5),(2,2),(2,3),
                     (2,4),(2,5),(1,2),(1,3),(1,4),(1,5),(2,0),(2,7),(3,0),(3,7),
                     (4,0),(4,7),(5,2),(5,3),(5,4),(5,5),(2,1),(2,6),(3,1),(3,6),
                     (4,1),(4,6),(1,1),(1,6),(1,0),(1,7),(5,0),(5,7),(5,1),(5,6),
                     (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)]

CHECK_ORDER_BLACK = [(3,3),(4,3),(3,4),(4,4),(4,2),(4,5),(3,2),(3,5),(5,2),(5,3),
                     (5,4),(5,5),(6,2),(6,3),(6,4),(6,5),(2,2),(2,3),(2,4),(2,5),
                     (5,1),(5,6),(4,1),(4,6),(3,1),(3,6),(2,1),(2,6),(6,1),(6,6),
                     (5,0),(5,7),(6,0),(6,7),(4,0),(4,7),(3,0),(3,7),(2,0),(2,7),
                     (7,1),(7,2),(7,3),(7,4),(7,5),(7,6)]


PLACEMAP = {WHITE: PLACEMAP_WHITE, BLACK: PLACEMAP_BLACK}


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

    if return_rowcol: return x, y

    return state[x, y]

###############################################################################
    
def initTable():
     ZobristTable = np.empty((SIZE, SIZE, 5))
     for i in range(SIZE):
         for j in range(SIZE):
             for k in range(5):
                 ZobristTable[i,j,k] = random.randint(0,1e19)

     return ZobristTable

###############################################################################

def zorHash(state, table):
    value = 0
    for i in range(SIZE):
        for j in range(SIZE):
            if state[i, j] != UNOCC:
                piece = state[i, j]
                value = value^int(table[i, j, piece])


    return value

###############################################################################

def hashMove(hashValue, colour, action):
    originPos = action[0]
    targetPos = action[1]
    newHash = copy.copy(hashValue)
    newHash = newHash^int(ZOR[originPos[0], originPos[1], colour])
    newHash = newHash^int(ZOR[targetPos[0], targetPos[1], colour])
    return newHash

###############################################################################

def hashRemove(hashValue, colour, position):
    newHash = copy.copy(hashValue)
    newHash = newHash^int(ZOR[position[0], position[1], colour])
    return newHash
###############################################################################
ZOR = initTable()