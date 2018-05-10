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

PHASE1 = 23
PHASE2 = 129
PHASE3 = 193

WIN = 9999
LOSE = -1 * WIN
TIE = 1000

weigt1 = 0
WEIGHTS = [165.59482,47.31155,117.44158,86.41152,110.65289]

IDEAL_DEPTH = {96:2,95:2,94:2,93:2,92:2,91:2,90:2,89:2,88:2,87:2,86:2,86:2,
               86:2,85:2,84:2,83:2,82:2,81:2,80:2,79:2,78:2,77:2,76:2,75:2,
               74:2,73:2,72:2,71:2,70:2,69:2,68:2,67:2,66:2,65:3,64:3,63:3,
               62:3,61:3,60:3,59:3,58:3,57:3,56:3,55:3,54:3,53:3,52:3,51:3,
               50:3,49:2,48:2,47:2,46:2,45:3,44:3,43:3,42:3,41:3,40:3,39:3,
               38:3,37:3,36:3,35:3,34:3,33:3,32:3,31:3,30:4,29:4,28:4,27:4,
               26:4,25:4,24:4,23:4,22:4,21:4,20:4,19:5,18:5,17:5,16:5,15:5,
               14:6,13:6,12:6,11:6,10:6,9:6,8:6,7:6,6:7,5:7,4:8,3:9,2:10,1:30,
               0:30}

PLACEMAP_WHITE = [[0,0,0,0,0,0,0,0],
                  [0,0,1,1,1,1,0,0],
                  [0,1,2,2,2,2,1,0],
                  [0,1,3,4,4,3,1,0],
                  [0,1,3,4,4,3,1,0],
                  [0,1,1,1,1,1,1,0],
                  [0,0,1,1,1,1,0,0],
                  [0,0,0,0,0,0,0,0]]

PLACEMAP_BLACK = [[0,0,0,0,0,0,0,0],
                  [0,0,1,1,1,1,0,0],
                  [0,1,1,1,1,1,1,0],
                  [0,1,3,4,4,3,1,0],
                  [0,1,3,4,4,3,1,0],
                  [0,1,2,2,2,2,1,0],
                  [0,0,1,1,1,1,0,0],
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

MAP = {WHITE:BLACK, BLACK:WHITE}
DEATHMAP = {WHITE: [6, 7], BLACK: [0, 1]}
PLACEMAP = {WHITE: PLACEMAP_WHITE, BLACK: PLACEMAP_BLACK}
ORDERMAP = {WHITE: ORDERMAP_WHITE, BLACK: ORDERMAP_BLACK}


MODS = {'R' :(0, 1),  # how each direction modifies a position
        '2R':(0, 2),
        'L' :(0, -1),
        '2L':(0, -2),
        'D' :(1, 0),
        '2D':(2, 0),
        'U' :(-1, 0),
        '2U':(-2, 0),
        'N' :(0, 0)}

###############################################################################

def pos_check(state, row, col, dir, return_rowcol=False):
    '''
    returns symbol at a given board position (modified by direction),
    optionally returns modified board coordinates
    '''
    x, y = row + MODS[dir][0], col + MODS[dir][1]

    if return_rowcol:
        return x, y

    return state[x, y]

###############################################################################

def init_table():
     zob_table = np.empty((SIZE, SIZE, 5))

     for i in range(SIZE):
         for j in range(SIZE):
             for k in range(5):
                 zob_table[i,j,k] = random.randint(0,1e19)

     return zob_table

###############################################################################

def zor_hash(state, table):
    value = 0

    for i in range(SIZE):
        for j in range(SIZE):
            if state[i, j] != UNOCC:
                piece = state[i, j]
                value = value^int(table[i, j, piece])


    return value

###############################################################################

def hash_mv(hash_val, colour, action):
    origin_pos = action[0]
    target_pos = action[1]
    new_hash = copy.copy(hash_val)
    new_hash = new_hash^int(ZOR[origin_pos[0], origin_pos[1], colour])
    new_hash = new_hash^int(ZOR[target_pos[0], target_pos[1], colour])

    return new_hash

###############################################################################

def hash_rm(hash_val, colour, position):
    new_hash = copy.copy(hash_val)
    new_hash = new_hash^int(ZOR[position[0], position[1], colour])

    return new_hash

###############################################################################

ZOR = init_table()


class Player():

    def __init__(self, colour):
        self.state = np.full((SIZE, SIZE), UNOCC, dtype=int)
        self.state[0,0] = CORNER
        self.state[0,7] = CORNER
        self.state[7,0] = CORNER
        self.state[7,7] = CORNER

        self.turns = 0
        self.total_turns = 0
        self.hash_table = dict()
        self.ab_hash = dict()
        self.visited = 0


        if colour[0] == 'w':
          self.player_colour = WHITE
          self.opp_colour = BLACK
          self.node = board(self.state, None, WHITE)
          self.place_moves = [(3,3),(4,3),(3,4),(4,4),(3,2),(3,5),(4,2),(4,5),
                              (2,2),(2,3),(2,4),(2,5),(5,2),(5,3),(5,4),(5,5),
                              (5,6),(1,3),(1,4),(1,5),(3,1),(3,6),(4,1),(4,6),
                              (5,1),(2,1),(2,6)]

        else:
          self.player_colour = BLACK
          self.opp_colour = WHITE
          self.node = board(self.state, None, BLACK)
          # finish place_moves for BLACK
          self.place_moves = [(4,4),(3,3),(4,3),(3,4),(5,5),(5,2),(4,2),(4,5),
                              (5,3),(5,4),(3,2),(3,5),(6,2),(6,3),(6,4),(6,5),
                              (5,6),(5,1),(4,1),(6,1),(6,6)]


###############################################################################

    def action(self, turns):
        if self.player_colour == WHITE:
            if turns == 128:
                # self.firstShrink(self.node)
                # print("i hope this is proc")
                self.shrink_board(self.node, 1)
                self.node.shrink_eliminate(1)

            if turns == 192:
                # self.secondShrink(self.node)
                self.shrink_board(self.node, 2)
                self.node.shrink_eliminate(2)

        

        # This is only used by player pieces
        self.turns = turns

        if self.total_turns > PHASE1:

            child_nodes_friendly = self.node.gen_child(self.player_colour)
            child_nodes_enemy = self.node.gen_child(self.opp_colour)

            total_branching = (len(child_nodes_friendly) +
                                len(child_nodes_enemy))
            action = self.mini_max(IDEAL_DEPTH[total_branching],
                                   child_nodes_friendly)

            self.total_turns += 1
            self.node.update_board_inplace(action, self.player_colour)

            if action == None:
                return None

            return (action[0][::-1], action[1][::-1])

        else:
            self.total_turns += 1
            place_move = self.mini_max_place(3)
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
            if (self.state[self.place_moves[i][0],
                           self.place_moves[i][1]] == UNOCC and
                not self.node.is_eliminated(self.state,
                                            self.place_moves[i][0],
                                            self.place_moves[i][1],
                                            self.player_colour)):

                return self.place_moves[i]

###############################################################################

    # This is only called by enemy pieces
    def update(self, action):
        size = np.array(action).size
        if self.player_colour == BLACK:
            if self.turns == 127:
                # self.firstShrink(self.node)
                # print("i hope this is proc")
                self.shrink_board(self.node, 1)
                self.node.shrink_eliminate(1)

            if self.turns == 191:
                # self.secondShrink(self.node)
                self.shrink_board(self.node, 2)
                self.node.shrink_eliminate(2)
        if size == 2:
            action = action[::-1]
            self.node.update_board_inplace(action, self.opp_colour)

        if size == 4:
            invert1 = action[0][::-1]
            invert2 = action[1][::-1]
            self.node.update_board_inplace((invert1, invert2), self.opp_colour)

        self.total_turns += 1

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

    def mini_max(self, depth, child):
        # start = time.time()
        current_hash = zor_hash(self.node.state, ZOR)

        def max_val(node_info, depth, alpha, beta, turns, hash_val):
            node = node_info[0]
            killed = node_info[1]

            if turns == PHASE2:
                self.shrink_board(node, 1)
                node.shrink_eliminate(1)
                node_hash = zor_hash(node.state, ZOR)

            elif turns == PHASE3:
                self.shrink_board(node, 2)
                node.shrink_eliminate(2)
                node_hash = zor_hash(node.state, ZOR)

            else:
                node_hash = hash_mv(hash_val, self.opp_colour, node.move)
                for dead in killed:
                    node_hash = hash_rm(node_hash, dead[1], dead[0])

            if node_hash in self.hash_table:
                node_val = self.hash_table[node_hash]
                self.visited+=1

                if node_hash in self.ab_hash:
                    if depth == self.ab_hash[node_hash][1]:
                        alpha, beta = self.ab_hash[node_hash][0]

            else:
                node_val = node.eval_func(2)
                self.hash_table[node_hash] = node_val

                if alpha != -np.inf and beta != np.inf:
                    self.ab_hash[node_hash] = ((alpha, beta), depth)

            if depth <= 0 or node_val in {WIN, LOSE, TIE}:
                return node_val

            v = -np.inf
            ordered_child_nodes = sorted(node.gen_child(node.colour),
                                         key=lambda x: x[0].move_estim,
                                         reverse=True)

            for child in ordered_child_nodes:
                v = max(v, min_val(child, depth-1, alpha, beta, turns+1,
                                                                node_hash))
                if v >= beta:
                    return v

                alpha = max(alpha, v)

            return v


        def min_val(node_info, depth, alpha, beta, turns, hash_val):
            node = node_info[0]
            killed = node_info[1]

            if turns == PHASE2:
                self.shrink_board(node, 1)
                node.shrink_eliminate(1)
                node_hash = zor_hash(node.state, ZOR)

            elif turns == PHASE3:
                self.shrink_board(node, 2)
                node.shrink_eliminate(2)
                node_hash = zor_hash(node.state, ZOR)

            else:
                node_hash = hash_mv(hash_val, self.player_colour, node.move)
                for dead in killed:
                    node_hash = hash_rm(node_hash, dead[1], dead[0])

            if node_hash in self.hash_table:
                node_val = self.hash_table[node_hash]
                self.visited += 1

                if node_hash in self.ab_hash:
                    if depth == self.ab_hash[node_hash][1]:
                        alpha, beta = self.ab_hash[node_hash][0]

            else:
                node_val = node.eval_func(2)
                self.hash_table[node_hash] = node_val
                if alpha != -np.inf and beta != np.inf:
                    self.ab_hash[node_hash] = ((alpha, beta), depth)

            if depth <= 0 or node_val in {WIN, LOSE, TIE}:
                return node_val

            v = np.inf
            ordered_child_nodes = sorted(node.gen_child(MAP[node.colour]),
                                         key=lambda x: x[0].move_estim)

            for child in ordered_child_nodes:
                v = min(v, max_val(child, depth-1, alpha, beta, turns+1,
                                                                node_hash))
                if v <= alpha:
                    return v

                beta = min(beta, v)

            return v

        best_score = -np.inf
        beta = np.inf
        best_action = None
        child_nodes = sorted(child, key=lambda x: x[0].move_estim,
                                    reverse=True)

        for child in child_nodes:
            v = min_val(child, depth-1, best_score, beta, self.turns,
                                                          current_hash)
            if v > best_score:
                best_score = v
                best_action = child[0].move

        # end = time.time()
        # print(end - start)
        return best_action

###############################################################################

    def mini_max_place(self, depth):
        # start = time.time()
        # print(depth)

        def max_val(node_info, depth, alpha, beta):
            node = node_info[0]
            node_val = node.eval_func(1)

            if depth <= 0 or node_val in {WIN, LOSE, TIE}:
                return node_val

            v = -np.inf
            ordered_child_nodes = sorted(node.gen_child_place_agr(node.colour),
                                         key=lambda x: x[0].move_estim,
                                         reverse=True)

            for child in ordered_child_nodes:
                v = max(v, min_val(child, depth-1, alpha, beta))

                if v >= beta:
                    return v

                alpha = max(alpha, v)

            return v


        def min_val(node_info, depth, alpha, beta):
            node = node_info[0]

            node_val = node.eval_func(1)

            if depth <= 0 or node_val in {WIN, LOSE, TIE}:
                return node_val

            v = np.inf
            ordered_child_nodes = node.gen_child_place_agr(MAP[node.colour])
            ordered_child_nodes = sorted(ordered_child_nodes,
                                         key=lambda x: x[0].move_estim)

            for child in ordered_child_nodes:
                v = min(v, max_val(child, depth-1, alpha, beta))

                if v <= alpha:
                    return v

                beta = min(beta, v)

            return v

        best_score = -np.inf
        beta = np.inf
        best_action = None

        for child in self.node.gen_child_place(self.player_colour):
            v = min_val(child, depth-1, best_score, beta)

            if v > best_score:
                best_score = v
                best_action = child[0].move

        # end = time.time()
        # print(end - start)
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
        new_state = np.copy(self.state)
        action_tuple = np.array(action)
        action_size = action_tuple.size

        if action_size == 1:
          return

        elif action_size == 2:
          #placing phase
          new_state[action_tuple[0], action_tuple[1]] = colour

        elif action_size == 4:
          # moving phase
          colour = self.state[action[0][0], action[0][1]]
          new_state[action_tuple[0][0], action_tuple[0][1]] = UNOCC
          new_state[action_tuple[1][0], action_tuple[1][1]] = colour

        elim_pieces = self.eliminate_board(new_state, colour)

        return board(new_state, action, self.colour), elim_pieces

###############################################################################

    def is_eliminated(self, board, row, col, piece):
        '''
        check whether the given piece will be eliminated by the corner
            and/or surrounding opponents
        '''
        opp_colour = MAP[piece]

        if row == 0 or row == 7:
            check_left = pos_check(board, row, col, 'L')
            check_right = pos_check(board, row, col, 'R')
            if check_left == opp_colour or check_left == CORNER:
                if check_right == opp_colour or check_right == CORNER:
                    return True

        elif col == 0 or col == 7:
            check_up = pos_check(board, row, col, 'U')
            check_down = pos_check(board, row, col, 'D')
            if check_up == opp_colour or check_up == CORNER:
                if check_down == opp_colour or check_down == CORNER:
                    return True

        else:
            # generate positions to check
            check = [pos_check(board,row,col,i) for i in ['L','R','U','D']]
            if check[0] == opp_colour or check[0] == CORNER:
                if check[1] == opp_colour or check[1] == CORNER:
                    return True
            if check[2] == opp_colour or check[2] == CORNER:
                if check[3] == opp_colour or check[3] == CORNER:
                    return True

        return False

###############################################################################

    def eliminate_board(self, state, colour):
        '''
        updates board and returns eliminated pieces
        '''
        eliminated = []
        elim_order = {WHITE: [BLACK, WHITE], BLACK: [WHITE, BLACK]}
        # order of elimination

        for piece in elim_order[colour]:
            for row, line in enumerate(state):
                for col, symbol in enumerate(line):
                    if symbol == piece:
                        if self.is_eliminated(state, row, col, piece):
                            state[row][col] = UNOCC
                            eliminated.append(((row, col), piece))

        return eliminated

###############################################################################

    def shrink_eliminate(self, shrink):
        '''
        elimination on board shrink as per forum post
        '''

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

        if action_size == 1:
            return

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
        legal_moves = 0
        check_dir = {'D':[row+1 < SIZE, row+2 < SIZE],
                     'U':[row-1 >= 0, row-2 >= 0],
                     'R':[col+1 < SIZE, col+2 < SIZE],
                     'L':[col-1 >= 0, col-2 >= 0]}

        for m in check_dir:
            if check_dir[m][0]:
                row2, col2 = pos_check(board, row, col, m, return_rowcol=True)
                new_pos = self.state[row2,col2]

                if new_pos == UNOCC and not self.is_eliminated(board, row2,
                                                               col2, piece):
                    legal_moves += 1

                if new_pos == WHITE or new_pos == BLACK:
                    if check_dir[m][1]:
                        row3, col3 = pos_check(board, row, col, '2' + m,
                                               return_rowcol=True)
                        if (self.state[row3,col3] == UNOCC and not
                            self.is_eliminated(board, row3, col3, piece)):
                            legal_moves += 1

        return legal_moves

###############################################################################

    def eval_func(self, phase):
        opp_colour = MAP[self.colour]
        results = np.bincount(self.state.ravel())

        # simple piece counter
        f1 = results[self.colour]
        f2 = results[opp_colour]
        f3 = 0  # board val
        f4 = 0  # safe mob
        f5 = 0  # connectdness


        if phase == 1:  #placing
            # center control + connectedness
            for row, line in enumerate(self.state):
                for col, symbol in enumerate(line):
                    if symbol  == self.colour:
                        f3 += PLACEMAP[self.colour][row][col]
                        check_dir = {'D':row+1 < SIZE,
                                     'U':row-1 >= 0,
                                     'R':col+1 < SIZE,
                                     'L':col-1 >= 0}

                        for m in check_dir:
                            if check_dir[m]:
                                if (pos_check(self.state, row, col, m) ==
                                    self.colour):
                                    f5 += 1


        if phase == 2:  #all moving phases
            # safe mobility + center control
            if results[self.colour] < 2 and results[opp_colour] >= 2:
                return LOSE
            if results[self.colour] >= 2 and results[opp_colour] < 2:
                return WIN
            if results[self.colour] < 2 and results[opp_colour] < 2:
                return TIE

            for row, line in enumerate(self.state):
                for col, symbol in enumerate(line):
                    if symbol == self.colour:
                        f3 += PLACEMAP[self.colour][row][col]   # placements scoring
                        f4 += self.count_legal_moves(self.state, row , col,
                                                     self.colour) # safe mobility
                        check_dir = {'D':row+1 < SIZE,
                                     'U':row-1 >= 0,
                                     'R':col+1 < SIZE,
                                     'L':col-1 >= 0}

                        for m in check_dir:
                            if check_dir[m]:
                                if (pos_check(self.state, row, col, m) ==
                                    self.colour):
                                    f5 += 1     # surrounding friendly pieces


        return (f1*WEIGHTS[0] + f2*WEIGHTS[1] + f3*WEIGHTS[2] + f4*WEIGHTS[3]
                    + f5*WEIGHTS[4])

###############################################################################

    def gen_child(self, colour):
        child_nodes = []
        action_tuple = ()

        for row, line in enumerate(self.state):
            # describe up/down moves to check
            check_dir = {'D': [row+1 < SIZE, 1, 0, row+2 < SIZE, 2, 0],
                         'U': [row-1 >= 0, -1, 0, row-2 >= 0, -2, 0]}

            for col, symbol in enumerate(line):
                # describe left/right moves to check
                check_dir['R'] = [col+1 < SIZE, 0, 1, col+2 < SIZE, 0, 2]
                check_dir['L'] = [col-1 >= 0, 0, -1, col-2 >= 0, 0, -2]

                if symbol == colour:
                    for dir in check_dir:
                        if check_dir[dir][0]:
                            to_check = pos_check(self.state, row, col, dir)

                            if to_check == UNOCC:
                                tmp_a = row + check_dir[dir][1]
                                tmp_b = col + check_dir[dir][2]
                                action_tuple = ((row, col), (tmp_a, tmp_b))
                                (child_nodes.append(self.update_board_return(
                                                        action_tuple, colour)))

                            elif to_check in {WHITE, BLACK}:
                                # check whether jump is possible
                                if check_dir[dir][3]:
                                    j = '2' + dir  # jump direction
                                    if (pos_check(self.state,row,col,j) ==
                                                                        UNOCC):

                                        tmp_a = row + check_dir[dir][4]
                                        tmp_b = col + check_dir[dir][5]
                                        action_tuple = ((row, col), (tmp_a,
                                                                     tmp_b))
                                        (child_nodes.append(
                                            self.update_board_return(
                                                action_tuple, colour)))

        return child_nodes


###############################################################################

    def gen_child_place(self, colour):
        child_nodes = []

        for row, line in enumerate(self.state):
            for col, symbol in enumerate(line):
                if row not in DEATHMAP[colour]:
                    if self.state[row, col] == UNOCC:
                        child_nodes.append(self.update_board_return((row, col),
                                                                     colour))

        return child_nodes

###############################################################################

    # This aggressively prunes by only only expanding nodes in a particular
    # order
    def gen_child_place_agr(self, colour):
        child_nodes = []
        board_val = ORDERMAP[colour]

        for placements in board_val:
            if placements[0] not in DEATHMAP[colour]:
                if self.state[placements[0], placements[1]] == UNOCC:
                    (child_nodes.append(self.update_board_return(
                        (placements[0], placements[1]), colour)))

        return child_nodes

###############################################################################
