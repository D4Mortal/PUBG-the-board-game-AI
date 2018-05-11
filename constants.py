# Chirag Rao Sahib      : 836011
# Daniel Hao            : 834496
# Date                  : 10/05/2018
# Python version        : 3.6.4

# Module                : constants.py

###############################################################################

SIZE = 8  # board size

UNOCC = 0  # = -
WHITE = 1  # = O
BLACK = 2  # = @
CORNER = 3  # = X
WALL = 4  # during shrinking

PHASE1 = 23
PHASE2 = 129
PHASE3 = 193

WIN = 9999
LOSE = -WIN
TIE = 1000
WEIGHTS = [1000, 5, 0.2, 2]  # eval_func weights

IDEAL_DEPTH = {96:2,95:2,94:2,93:2,92:2,91:2,90:2,89:2,88:2,87:2,86:2,86:2,
               86:2,85:2,84:2,83:2,82:2,81:2,80:2,79:2,78:2,77:2,76:2,75:2,
               74:2,73:2,72:2,71:2,70:2,69:2,68:2,67:2,66:2,65:2,64:2,63:3,
               62:3,61:3,60:3,59:3,58:3,57:3,56:3,55:3,54:3,53:3,52:3,51:3,
               50:3,49:3,48:3,47:2,46:2,45:3,44:3,43:3,42:3,41:3,40:3,39:3,
               38:3,37:3,36:3,35:3,34:4,33:4,32:4,31:4,30:4,29:4,28:4,27:4,
               26:4,25:4,24:4,23:4,22:4,21:5,20:5,19:5,18:5,17:5,16:5,15:5,
               14:6,13:6,12:6,11:6,10:6,9:6,8:6,7:6,6:7,5:7,4:8,3:9,2:10,1:30,
               0:30}

PLACEMAP_WHITE = [[0,0,0,0,0,0,0,0],
                  [0,0,0,1,1,0,0,0],
                  [0,0,2,2,2,2,0,0],
                  [0,1,3,4,4,3,1,0],
                  [0,1,3,4,4,3,1,0],
                  [0,0,1,1,1,1,0,0],
                  [0,0,0,1,1,0,0,0],
                  [0,0,0,0,0,0,0,0]]

PLACEMAP_BLACK = [[0,0,0,0,0,0,0,0],
                  [0,0,0,1,1,0,0,0],
                  [0,0,1,1,1,1,0,0],
                  [0,1,3,4,4,3,1,0],
                  [0,1,3,4,4,3,1,0],
                  [0,0,2,2,2,2,0,0],
                  [0,0,0,1,1,0,0,0],
                  [0,0,0,0,0,0,0,0]]

ORDERMAP_WHITE = [(3,3),(4,3),(3,4),(4,4),(3,2),(3,5),(4,2),(4,5),(2,2),(2,3),
                  (2,4),(2,5),(1,2),(1,3),(1,4),(1,5),(2,0),(2,7),(3,0),(3,7),
                  (4,0),(4,7),(5,2),(5,3),(5,4),(5,5),(2,1),(2,6),(3,1),(3,6),
                  (4,1),(4,6),(1,1),(1,6),(1,0),(1,7),(5,0),(5,7),(5,1),(5,6),
                  (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)]

ORDERMAP_BLACK = [(3,3),(4,3),(3,4),(4,4),(4,2),(4,5),(3,2),(3,5),(5,2),(5,3),
                  (5,4),(5,5),(6,2),(6,3),(6,4),(6,5),(2,2),(2,3),(2,4),(2,5),
                  (5,1),(5,6),(4,1),(4,6),(3,1),(3,6),(2,1),(2,6),(6,1),(6,6),
                  (5,0),(5,7),(6,0),(6,7),(4,0),(4,7),(3,0),(3,7),(2,0),(2,7),
                  (7,1),(7,2),(7,3),(7,4),(7,5),(7,6)]


MAP = {WHITE:BLACK, BLACK:WHITE}  # map player->opponent colour
DEATHMAP = {WHITE: [6, 7], BLACK: [0, 1]}  # out of bounds rows (placing)

# piece-square table for each colour
PLACEMAP = {WHITE: PLACEMAP_WHITE, BLACK: PLACEMAP_BLACK}

# strict move ordering for alpha-beta (placing)
ORDERMAP = {WHITE: ORDERMAP_WHITE, BLACK: ORDERMAP_BLACK}

# direction modifiers
MODS = {'R' :(0, 1),
        '2R':(0, 2),
        'L' :(0, -1),
        '2L':(0, -2),
        'D' :(1, 0),
        '2D':(2, 0),
        'U' :(-1, 0),
        '2U':(-2, 0),
        'N' :(0, 0)}
