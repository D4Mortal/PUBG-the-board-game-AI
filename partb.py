import copy
import sys
import time
import random
import numpy as np


SIZE = 8  # board size

UNOCC = 0  #'-'
WHITE = 1  #'O'
BLACK = 2  #'@'
CORNER = 3  #'X'
WALL = 4

MINIMAX_DEPTH_1 = 2
MINIMAX_DEPTH_2 = 2
GO_HARD = 8  # change minimax depth player pieces

PHASE1 = 23
PHASE2 = PHASE1 + 128
PHASE3 = PHASE2 + 64

WIN = 9999
LOSE = -1 * WIN
TIE = 1000

WEIGHTS = [1000, 5, 0.2, 2]

IDEAL_DEPTH = {80:2,79:2,78:2,77:2,76:2,75:2,74:2,73:3,72:3,71:3,70:3,69:3,68:3,
               67:3,66:3,65:3,64:3,63:3,62:3,61:3,60:3,59:3,58:3,57:3,56:3,55:3,
               54:3,53:3,52:3,51:3,50:3,49:3,48:3,47:3,46:3,45:3,44:3,43:3,42:3,
               41:3,40:3,39:3,38:3,37:3,36:4,35:3,34:4,33:4,32:4,31:4,30:4,29:3,
               28:4,27:4,26:4,25:4,24:5,23:4,22:5,21:5,20:4,19:5,18:5,17:4,16:5,
               15:5,14:6,13:5,12:5,11:6,10:6,9:6,8:6,7:6,6:6,5:6,4:6,3:6,2:6}

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

PLACEMAP_WHITE2 = [[-1,-1,-1,-1,-1,-1,-1,-1],    
                  [-1,-1,0,0,0,0,-1,-1],
                  [-1,0,1,2,2,1,0,-1],
                  [-1,0,2,4,4,2,0,-1],
                  [-1,0,3,4,4,3,0,-1],
                  [-1,0,0,1,1,0,0,-1],
                  [-1,-1,0,0,0,0,-1,-1],
                  [-1,-1,-1,-1,-1,-1,-1,-1]]

PLACEMAP_BLACK2 =  [[-1,-1,-1,-1,-1,-1,-1,-1],    
                  [-1,-1,0,0,0,0,-1,-1],
                  [-1,0,0,1,1,0,0,-1],
                  [-1,0,3,4,4,3,0,-1],
                  [-1,0,2,4,4,2,0,-1],
                  [-1,0,1,2,2,1,0,-1],
                  [-1,-1,0,0,0,0,-1,-1],
                  [-1,-1,-1,-1,-1,-1,-1,-1]]

CHECK_ORDER_WHITE = [(3,3),(4,3),(3,4),(4,4),(3,2),(3,5),(4,2),(4,5),(2,2),(2,3),(2,4),(2,5),
                     (1,2),(1,3),(1,4),(1,5),(2,0),(2,7),
                     (3,0),(3,7),(4,0),(4,7),
                     (5,2),(5,3),(5,4),(5,5),(2,1),(2,6),(3,1),(3,6),(4,1),(4,6),(1,1),(1,6),(1,0),(1,7),
                     (5,0),(5,7),(5,1),(5,6),(0,0),           
                     (0,1),(0,2),(0,3),(0,4),(0,5),(0,6)]

CHECK_ORDER_BLACK = [(3,3),(4,3),(3,4),(4,4),(4,2),(4,5),(3,2),(3,5),(5,2),(5,3),(5,4),
          (5,5),(6,2),(6,3),(6,4),(6,5),(2,2),(2,3),(2,4),(2,5),(5,1),(5,6),(4,1),(4,6),(3,1),
          (3,6),(2,1),(2,6),(6,1),(6,6),(5,0),(5,7),(6,0),(6,7),(4,0),(4,7),(3,0),(3,7),
          (2,0),(2,7),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6)]


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

class Player():

    def __init__(self, colour):
        self.state = np.full((SIZE, SIZE), UNOCC, dtype=int)
        self.state[0,0] = CORNER
        self.state[0,7] = CORNER
        self.state[7,0] = CORNER
        self.state[7,7] = CORNER

        self.turns = 0
        self.totalTurns = 0
        self.hashTable = dict()
        self.abHash = dict()
        self.visited = 0


        if colour[0] == 'w':
          self.player_colour = WHITE
          self.opp_colour = BLACK
          self.node = board(self.state, None, WHITE)
          self.place_moves = [(3,3),(4,3),(3,4),(4,4),(3,2),(3,5),(4,2),(4,5),
          (2,2),(2,3),(2,4),(2,5),(5,2),(5,3),(5,4),(5,5),
          (5,6), (1,3),(1,4),(1,5),(3,1),(3,6),(4,1),(4,6),(5,1),
          (2,1),(2,6)]

        else:
          self.player_colour = BLACK
          self.opp_colour = WHITE
          self.node = board(self.state, None, BLACK)
          # finish place_moves for BLACK
          self.place_moves = [(4,4), (3,3),(4,3),(3,4),(5,5),(5,2),(4,2),(4,5),
                              (5,3),(5,4),(3,2),(3,5),(6,2),(6,3),(6,4),(6,5),
                              (5,6),(5,1),(4,1),(6,1),(6,6)]


###############################################################################

    def action(self, turns):
        if turns == 128:
            # self.firstShrink(self.node)
            print("i hope this is proc")
            self.shrink_board(self.node, 1)
            self.node.shrink_eliminate(1)

        if turns == 192:
            # self.secondShrink(self.node)
            self.shrink_board(self.node, 2)
            self.node.shrink_eliminate(2)

        # This is only used by player pieces
        self.turns = turns + 1
        # print(self.node.state)

        if self.totalTurns > PHASE1:
           
            child_nodes_friendly = self.node.genChild(self.player_colour)
            child_nodes_enemy = self.node.genChild(self.opp_colour)
            
            total_branching = len(child_nodes_friendly) + len(child_nodes_enemy)
            
            action = self.miniMax(IDEAL_DEPTH[total_branching],child_nodes_friendly)
            
            self.totalTurns += 1
            self.node.update_board_inplace(action, self.player_colour)
            
            if action == None:
                return None
            return (action[0][::-1], action[1][::-1])

        else:
            self.totalTurns += 1
            place_move = self.miniMaxPlace(3)
            self.node.update_board_inplace(place_move, self.player_colour)

            return place_move[::-1]

###############################################################################

    def place_phase(self):
        danger_result = self.in_danger(self.player_colour)
        if danger_result != None:
            return danger_result

        kill_result = self.in_danger(self.opp_colour)
        if kill_result != None:
            return kill_result

        for i in range(len(self.place_moves)):
            if self.state[self.place_moves[i][0], self.place_moves[i][1]] == UNOCC and not self.node.is_eliminated(self.state, self.place_moves[i][0], self.place_moves[i][1], self.player_colour):
                return self.place_moves[i]

###############################################################################

    # This is only called by enemy pieces
    def update(self, action):
        action_tuple = np.array(action)
        size = action_tuple.size

        if size == 2:
            action = action[::-1]
            self.node.update_board_inplace(action, self.opp_colour)

        if size == 4:
            if self.turns == 128:
                # self.firstShrink(self.node)
                self.shrink_board(self.node, 1)
                # self.node.shrinkKill1()
                self.node.shrink_eliminate(1)
                print(self.node.state)
            if self.turns == 192:
                self.shrink_board(self.node, 2)
                # self.secondShrink(self.node)
                # self.node.shrinkKill2()
                self.node.shrink_eliminate(2)


            invert1 = action[0][::-1]
            invert2 = action[1][::-1]
            self.node.update_board_inplace((invert1, invert2), self.opp_colour)

        self.totalTurns += 1

###############################################################################

    def shrink_board(self, node, shrink):
        if shrink == 1:
            node.state[0, :] = WALL
            node.state[7, :] = WALL
            node.state[:, 0] = WALL
            node.state[:, 7] = WALL
            node.state[1,1] = CORNER
            node.state[1,6] = CORNER
            node.state[6,1] = CORNER
            node.state[6,6] = CORNER

        if shrink == 2:
            node.state[1, :] = WALL
            node.state[6, :] = WALL
            node.state[:, 1] = WALL
            node.state[:, 6] = WALL
            node.state[2,2] = CORNER
            node.state[2,5] = CORNER
            node.state[5,2] = CORNER
            node.state[5,5] = CORNER

###############################################################################

    def put_piece(self, row, col, piece):
      self.state[row, col] = piece

###############################################################################
    def miniMax(self, depth, child):
        start = time.time()
        currentHash = zorHash(self.node.state, ZOR)

        def maxValue(nodeInfo, depth, alpha, beta, turns, hashValue):
            node = nodeInfo[0]
            killed = nodeInfo[1]

            if turns == 129:
                self.shrink_board(node, 1)
                # node.shrinkKill1()
                node.shrink_eliminate(1)

                nodeHash = zorHash(node.state, ZOR)

            elif turns == 193:
                self.shrink_board(node, 2)
                # self.secondShrink(node)
                # node.shrinkKill2()
                node.shrink_eliminate(2)

                nodeHash = zorHash(node.state, ZOR)

            else:
                nodeHash = hashMove(hashValue, self.opp_colour, node.move)
                for dead in killed:
                    nodeHash = hashRemove(nodeHash, dead[1], dead[0])

            if nodeHash in self.hashTable:
                nodeValue = self.hashTable[nodeHash]
                self.visited+=1
                if nodeHash in self.abHash:
                    if turns == self.abHash[nodeHash][1]:
                        alpha, beta = self.abHash[nodeHash][0]

            else:
                nodeValue = node.eval_func(2)
                self.hashTable[nodeHash] = nodeValue
                if alpha != -np.inf and beta != np.inf:
                    self.abHash[nodeHash] = ((alpha, beta), turns)

            if  depth <= 0 or nodeValue == LOSE or nodeValue == WIN or nodeValue == TIE:
                return nodeValue

            v = -np.inf

            ordered_child_nodes = sorted(node.genChild(node.colour),
                key=lambda x: x[0].move_estim, reverse=True)

            for child in ordered_child_nodes:

                v = max(v, minValue(child, depth-1, alpha, beta, turns+1, nodeHash))
#                print(child.eval_node(), end='')
#                print("White's move: ", end='')
#                print(child.move)
#                print(child.state)
                if v >= beta: return v
                alpha = max(alpha, v)

            return v


        def minValue(nodeInfo, depth, alpha, beta, turns, hashValue):
            node = nodeInfo[0]
            killed = nodeInfo[1]

            if turns == 129:
                # self.firstShrink(node) self.shrink_board(node, 1)
                # node.shrinkKill1()
                node.shrink_eliminate(1)
                nodeHash = zorHash(node.state, ZOR)

            elif turns == 193:
                # self.secondShrink(node)
                self.shrink_board(node, 2)
                # node.shrinkKill2()
                node.shrink_eliminate(2)
                nodeHash = zorHash(node.state, ZOR)

            else:
                nodeHash = hashMove(hashValue, self.player_colour, node.move)
                for dead in killed:
                    nodeHash = hashRemove(nodeHash, dead[1], dead[0])

            if nodeHash in self.hashTable:
                nodeValue = self.hashTable[nodeHash]
                self.visited += 1
                if nodeHash in self.abHash:
                    if turns == self.abHash[nodeHash][1]:
                        alpha, beta = self.abHash[nodeHash][0]

            else:
                nodeValue = node.eval_func(2)
                self.hashTable[nodeHash] = nodeValue
                if alpha != -np.inf and beta != np.inf:
                    self.abHash[nodeHash] = ((alpha, beta), turns)

            if  depth <= 0 or nodeValue == LOSE or nodeValue == WIN or nodeValue == TIE:
                return nodeValue

            v = np.inf
            ordered_child_nodes = sorted(node.genChild(MAP[node.colour]),
                key=lambda x: x[0].move_estim)

            for child in ordered_child_nodes:
                v = min(v, maxValue(child, depth-1, alpha, beta, turns+1, nodeHash))
#                print(child.eval_node(), end='')
#                print(child.state)
                if v <= alpha:
                    return v
                beta = min(beta, v)

            return v

        best_score = -np.inf
        beta = np.inf
        best_action = None
        
        child_nodes = sorted(child, key=lambda x: x[0].move_estim, reverse=True)
        for child in child_nodes:
            v = minValue(child, depth-1, best_score, beta, self.turns, currentHash)
            if v > best_score:
                best_score = v
                best_action = child[0].move

        end = time.time()
        print(end - start)
        return best_action

###############################################################################

    def miniMaxPlace(self, depth):
        start = time.time()
        print(depth)

        def maxValue(nodeInfo, depth, alpha, beta):
            node = nodeInfo[0]

            nodeValue = node.eval_func(1)
            if  depth <= 0 or nodeValue == LOSE or nodeValue == WIN or nodeValue == TIE:
                return nodeValue

            v = -np.inf

            ordered_child_nodes = sorted(node.genChildPlaceAgressive(node.colour),
                key=lambda x: x[0].move_estim, reverse=True)

            for child in ordered_child_nodes:
                v = max(v, minValue(child, depth-1, alpha, beta))
#                print(child.eval_node(), end='')
#                print("White's move: ", end='')
#                print(child.move)
#                print(child.state)
                if v >= beta: return v
                alpha = max(alpha, v)

            return v


        def minValue(nodeInfo, depth, alpha, beta):
            node = nodeInfo[0]

            nodeValue = node.eval_func(1)

            if  depth <= 0 or nodeValue == LOSE or nodeValue == WIN or nodeValue == TIE:
                return nodeValue

            v = np.inf
            ordered_child_nodes = sorted(node.genChildPlaceAgressive(MAP[node.colour]),
                key=lambda x: x[0].move_estim)

            for child in ordered_child_nodes:
                v = min(v, maxValue(child, depth-1, alpha, beta))
#                print(child.eval_node(), end='')
#                print(child.state)
                if v <= alpha:
                    return v
                beta = min(beta, v)

            return v

        best_score = -np.inf
        beta = np.inf
        best_action = None

        for child in self.node.genChildPlace(self.player_colour):
            v = minValue(child, depth-1, best_score, beta)
            if v > best_score:
                best_score = v
                best_action = child[0].move

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
    def update_board_return(self, action, colour):
        newState = np.copy(self.state)
        action_tuple = np.array(action)
        action_size = action_tuple.size

        if action_size == 1:
          return

        elif action_size == 2:
          #placing phase

          newState[action_tuple[0], action_tuple[1]] = colour

        elif action_size == 4:
          # moving phase
          colour = self.state[action[0][0], action[0][1]]
          newState[action_tuple[0][0], action_tuple[0][1]] = UNOCC
          newState[action_tuple[1][0], action_tuple[1][1]] = colour

        eleminated = self.eliminate_board(newState, colour)

        return board(newState, action, self.colour), eleminated

###############################################################################

    def is_eliminated(self, board, row, col, piece):
        '''
        check whether the given piece will be eliminated by the corner
            and/or surrounding opponents
        '''
        Opp = MAP[piece]
        if row == 0 or row == 7:
            checkLeft = pos_check(board, row, col, 'L')
            checkRight = pos_check(board, row, col, 'R')
            if checkLeft == Opp or checkLeft == CORNER:
                if checkRight == Opp or checkRight == CORNER:
                    return True

        elif col == 0 or col == 7:
            checkUp = pos_check(board, row, col, 'U')
            checkDown = pos_check(board, row, col, 'D')
            if checkUp == Opp or checkUp == CORNER:
                if checkDown == Opp or checkDown == CORNER:
                    return True

        else:
            # generate positions to check
            check = [pos_check(board,row,col,i) for i in ['L','R','U','D']]
            if check[0] == Opp or check[0] == CORNER:
                if check[1] == Opp or check[1] == CORNER:
                    return True
            if check[2] == Opp or check[2] == CORNER:
                if check[3] == Opp or check[3] == CORNER:
                    return True

        return False

###############################################################################

    def eliminate_board(self, state, colour):
        '''
        returns updated board after necessary eliminations
        '''
        eliminated = []
        mapping = {WHITE: [BLACK, WHITE], BLACK: [WHITE, BLACK]}
        # order of elimination

        for piece in mapping[colour]:
            for row, line in enumerate(state):
                for col, symbol in enumerate(line):
                    if symbol == piece:
                        if self.is_eliminated(state, row, col, piece):
                            state[row][col] = UNOCC
                            eliminated.append(((row, col), piece))
        return eliminated

###############################################################################

    def shrink_eliminate(self, shrink):

        if shrink == 1:
            if self.state[1, 2] != UNOCC and self.state[1, 3] != UNOCC:
                if self.state[1, 2] != self.state[1, 3]:
                    self.state[1, 2] = UNOCC

            if self.state[1, 4] != UNOCC and self.state[1, 4] != UNOCC:
                if self.state[1, 4] != self.state[1, 4]:
                    self.state[1, 5] = UNOCC

            if self.state[2, 1] != UNOCC and self.state[3, 1] != UNOCC:
                if self.state[2, 1] != self.state[3, 1]:
                    self.state[2, 1] = UNOCC

            if self.state[5, 1] != UNOCC and self.state[4, 1] != UNOCC:
                if self.state[5, 1] != self.state[4, 1]:
                    self.state[5, 1] = UNOCC

            if self.state[6, 2] != UNOCC and self.state[6, 3] != UNOCC:
                if self.state[6, 2] != self.state[6, 3]:
                    self.state[6, 2] = UNOCC

            if self.state[6, 5] != UNOCC and self.state[6, 4] != UNOCC:
                if self.state[6, 5] != self.state[6, 4]:
                    self.state[6, 5] = UNOCC

            if self.state[5, 6] != UNOCC and self.state[4, 6] != UNOCC:
                if self.state[5, 6] != self.state[4, 6]:
                    self.state[5, 6] = UNOCC

            if self.state[2, 6] != UNOCC and self.state[3, 6] != UNOCC:
                if self.state[2, 6] != self.state[3, 6]:
                    self.state[2, 6] = UNOCC

        if shrink == 2:
            if self.state[2, 3] != UNOCC and self.state[2, 4] != UNOCC:
                if self.state[2, 3] != self.state[2, 4]:
                    self.state[2, 3] = UNOCC

            if self.state[3, 2] != UNOCC and self.state[4, 2] != UNOCC:
                if self.state[3, 2] != self.state[4, 2]:
                    self.state[3, 2] = UNOCC

            if self.state[5, 3] != UNOCC and self.state[5, 4] != UNOCC:
                if self.state[5, 3] != self.state[5, 4]:
                    self.state[5, 3] = UNOCC

            if self.state[4, 5] != UNOCC and self.state[3, 5] != UNOCC:
                if self.state[4, 5] != self.state[3, 5]:
                    self.state[4, 5] = UNOCC

###############################################################################

    # function that make moves on the current object, changes the current state,
    # does not create a new board

    def update_board_inplace(self, action, colour):
        action_tuple = np.array(action)
        action_size = action_tuple.size

        if action_size == 1: return

        elif action_size == 2:
          #placing phase
          self.state[action_tuple[0], action_tuple[1]] = colour

        elif action_size == 4:
          # moving phase
          self.state[action_tuple[0][0], action_tuple[0][1]] = UNOCC
          self.state[action_tuple[1][0], action_tuple[1][1]] = colour


        self.eliminate_board(self.state, colour)
        self.move = action # update the move that brought it to this state

###############################################################################

    def pvs_estim(self):
        '''
        principal variation estimation function
        '''
        results = np.bincount(self.state.ravel())
        return results[self.colour] - results[MAP[self.colour]]

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

    def eval_func(self, phase):
        oppColour = MAP[self.colour]
        results = np.bincount(self.state.ravel())

        # simple piece counter
        f1 = results[self.colour] - results[MAP[self.colour]]
        f2 = 0
        f3 = 0
        f4 = 0  # connectdness

        if phase == 1:  #placing
            # center control + connectedness
            for row, line in enumerate(self.state):
                for col, symbol in enumerate(line):
                    if symbol  == self.colour:
                        f2 += PLACEMAP_WHITE[row][col]
                        checkCond = {'D':row+1 < SIZE,
                                 'U':row-1 >= 0,
                                 'R':col+1 < SIZE,
                                 'L':col-1 >= 0}

                        for m in checkCond:
                            if checkCond[m]:
                                if pos_check(self.state, row, col, m) == self.colour:
                                    f4 += 1


        if phase == 2:  #all moving phases
            # safe mobility + center control
            if results[self.colour] < 2 and results[oppColour] >= 2: return LOSE
            if results[self.colour] >= 2 and results[oppColour] < 2: return WIN
            if results[self.colour] < 2 and results[oppColour] < 2: return TIE

            for row, line in enumerate(self.state):
                for col, symbol in enumerate(line):
                    if symbol == self.colour:
                        f2 += PLACEMAP[self.colour][row][col]   # placements scoring
                        f3 += self.count_legal_moves(self.state, row , col, self.colour) # safe mobility
                        checkCond = {'D':row+1 < SIZE,
                                 'U':row-1 >= 0,
                                 'R':col+1 < SIZE,
                                 'L':col-1 >= 0}

                        for m in checkCond:
                            if checkCond[m]:    
                                if pos_check(self.state, row, col, m) == self.colour:
                                    f4 += 1     # surrounding friendly pieces


        return f1*WEIGHTS[0] + f2*WEIGHTS[1] + f3*WEIGHTS[2] + f4*WEIGHTS[3]

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
                                child_nodes.append(self.update_board_return(action_tuple, colour))

                            elif posToCheck == WHITE or posToCheck == BLACK:
                                # check whether jump is possible
                                if checkCond[dir][3]:
                                    j = '2' + dir  # jump direction
                                    if pos_check(self.state,row,col,j) == UNOCC:
                                        tmpA = row + checkCond[dir][4]
                                        tmpB = col + checkCond[dir][5]

                                        action_tuple = ((row, col), (tmpA, tmpB))
                                        child_nodes.append(self.update_board_return(action_tuple, colour))

        return child_nodes


###############################################################################

    def genChildPlace(self, colour):
        child_nodes = []

        for row, line in enumerate(self.state):
            for col, symbol in enumerate(line):
                if row not in DEATHMAP[colour]:
                    if self.state[row, col] == UNOCC:
                        child_nodes.append(self.update_board_return((row, col), colour))
        return child_nodes

###############################################################################  
    
    # This aggressively prunes by only only expanding 22 nodes, based on the assumption
    # that the opponent would try to play in the middle.
    def genChildPlaceAgressive(self, colour):
        child_nodes = []
        if colour == 1:
            mapping = CHECK_ORDER_WHITE
        elif colour == 2:
            mapping = CHECK_ORDER_BLACK
        
        for placements in mapping:

            if placements[0] not in DEATHMAP[colour]:
                if self.state[placements[0], placements[1]] == UNOCC:
                        child_nodes.append(self.update_board_return((placements[0], placements[1]), colour))

        
        return child_nodes    
    
ZOR = initTable()

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
    move = ((2,5), (2,6))
    move2 = ((2,4), (3,4))
    move3 = ((3,5), (1,1))
    move4 = ((6,6),(5,6))
    place = (6,5)
    null_move = None

#    print('before update')
    game.put_piece(4, 3, WHITE)  # example for move
    game.put_piece(4, 7, WHITE)  # example for move
    game.put_piece(2, 5, WHITE)  # example for move
    game.put_piece(4, 6, WHITE)  # example for move
    game.put_piece(1, 1, WHITE)  # example for move
    game.put_piece(0, 3, WHITE)  # example for move
    game.put_piece(2, 0, WHITE)  # example for move
    game.put_piece(6, 3, WHITE)  # example for move
    game.put_piece(0, 5, WHITE)  # example for move
    game.put_piece(5, 0, WHITE)  # example for move
    game.put_piece(4, 1, WHITE)  # example for move
    game.put_piece(6, 7, WHITE)  # example for move


    game.put_piece(2, 4, BLACK)  # example for move
    game.put_piece(2, 2, BLACK)  # example for move
    game.put_piece(3, 5, BLACK)  # example for move
    game.put_piece(3, 6, BLACK)  # example for move
    game.put_piece(3, 1, BLACK)  # example for move
    game.put_piece(3, 3, BLACK)  # example for move
    game.put_piece(5, 4, BLACK)  # example for move
    game.put_piece(3, 5, BLACK)  # example for move
    game.put_piece(0, 1, BLACK)  # example for move
    game.put_piece(2, 7, BLACK)  # example for move
    game.put_piece(7, 1, BLACK)  # example for move
    game.put_piece(7, 4, BLACK)  # example for move
    game.put_piece(6, 6, BLACK)  # example for move
#    print(game.node.state)
#    print(game.player_colour)
#    print(game.node.eval_node())
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
    # print(sys.getsizeof(game.hashTable))
    # print(game.visited)
    # print(game.miniMaxPlace(4))

# # #
#    print(sys.getsizeof(game.hashTable))

    # print("this is the current board state")
    # print(game.node.state)

#    print('place test')
#
#    for i in list(range(0,24,2)):
#        print('game move', i)
#        if i == 12:
#            game.put_piece(2, 4, BLACK)
#            game.put_piece(2,5, WHITE)
#        print('total_turns', game.totalTurns)
#        print(game.state)
#        game.action(i)
#    print("The ideal move would be: {} for turn 127".format(game.node.move))


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


#    r = zorHash(game.node.state, ZOR)
#    print(r)
#    print(hashMove(r, game.node.state, ((3,5), (1,1))))
#
#
#    game.update(move3)              # move3 is ((3,5), (1,1))
#    a = zorHash(game.node.state, ZOR)
#    print(a)
#
#    game.node.state[4,6] = UNOCC      # remove white piece and recalculate hash from scratch
#    a = zorHash(game.node.state, ZOR)
#    print(a)
#
#    r = r^int(ZOR[3, 5, BLACK])     # hash the moves ((3,5), (1,1)) from initial state
#    r = r^int(ZOR[1, 1, BLACK])
#
#    game.node.state[4,6] = WHITE    # put the white piece back in
#    print(hashRemove(r, game.node.state, (4,6))) # compute hash value from removing white piece





#testrun()
#testMemUsage()
