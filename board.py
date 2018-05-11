# Chirag Rao Sahib      : 836011
# Daniel Hao            : 834496
# Date                  : 10/05/2018
# Python version        : 3.6.4

# Module                : board.py

###############################################################################

import copy
import numpy as np
from constants import *

###############################################################################

class board(object):
    '''
    stores the current board config and the move that brought it there
    '''
    __slots__ = ('state', 'move', 'colour', 'move_estim')

    def __init__(self, state, move, colour):
        self.state = state
        self.move = move
        self.colour = colour
        self.move_estim = self.pvs_estim()

###############################################################################

    def pos_check(self, state, row, col, dir, return_rowcol=False):
        '''
        returns symbol at a given board position (modified by direction),
            optionally returns modified board coordinates
        '''
        x, y = row + MODS[dir][0], col + MODS[dir][1]

        if return_rowcol:
            return x, y

        return state[x, y]

###############################################################################

    def update_board_return(self, action, colour):
        '''
        returns new board object resulting from specified action
        '''
        new_state = np.copy(self.state)
        action_tuple = np.array(action)
        action_size = action_tuple.size

        if action_size == 1:  # action is None
          return

        elif action_size == 2:  # placing phase
          new_state[action_tuple[0], action_tuple[1]] = colour

        elif action_size == 4:  # moving phase
          colour = self.state[action[0][0], action[0][1]] # ***
          new_state[action_tuple[0][0], action_tuple[0][1]] = UNOCC
          new_state[action_tuple[1][0], action_tuple[1][1]] = colour

        elim_pieces = self.eliminate_board(new_state, colour)

        return board(new_state, action, self.colour), elim_pieces

###############################################################################

    def is_eliminated(self, board, row, col, piece):
        '''
        check whether the given piece will be eliminated by the corner
            and/or surrounding opponents (from parta)
        '''
        opp_colour = MAP[piece]

        if row == 0 or row == 7:
            check_left = self.pos_check(board, row, col, 'L')
            check_right = self.pos_check(board, row, col, 'R')
            if check_left == opp_colour or check_left == CORNER:
                if check_right == opp_colour or check_right == CORNER:
                    return True

        elif col == 0 or col == 7:
            check_up = self.pos_check(board, row, col, 'U')
            check_down = self.pos_check(board, row, col, 'D')
            if check_up == opp_colour or check_up == CORNER:
                if check_down == opp_colour or check_down == CORNER:
                    return True

        else:
            # generate positions to check
            check = ([self.pos_check(board,row,col,i) for i in
                      ['L','R','U','D']])
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
        updates board and returns eliminated pieces (from parta)
        '''
        eliminated = []

        # order of elimination
        elim_order = {WHITE: [BLACK, WHITE], BLACK: [WHITE, BLACK]}

        for piece in elim_order[colour]:
            for row, line in enumerate(state):
                for col, symbol in enumerate(line):
                    if symbol == piece:
                        if self.is_eliminated(state, row, col, piece):
                            state[row][col] = UNOCC
                            eliminated.append(((row, col), piece))

        return eliminated

###############################################################################

    def shrink_eliminate(self, shrink_no):
        '''
        anti-clockwise elimination on shrink as per forum post
        '''
        if shrink_no == 1:
            if (self.state[1, 2] != UNOCC and
                self.state[1, 3] != UNOCC and
                self.state[1, 2] != self.state[1, 3]):
                self.state[1, 2] = UNOCC

            if (self.state[1, 4] != UNOCC and
                self.state[1, 5] != UNOCC and
                self.state[1, 4] != self.state[1, 5]):
                self.state[1, 5] = UNOCC

            if (self.state[2, 1] != UNOCC and
                self.state[3, 1] != UNOCC and
                self.state[2, 1] != self.state[3, 1]):
                self.state[2, 1] = UNOCC

            if (self.state[5, 1] != UNOCC and
                self.state[4, 1] != UNOCC and
                self.state[5, 1] != self.state[4, 1]):
                self.state[5, 1] = UNOCC

            if (self.state[6, 2] != UNOCC and
                self.state[6, 3] != UNOCC and
                self.state[6, 2] != self.state[6, 3]):
                self.state[6, 2] = UNOCC

            if (self.state[6, 5] != UNOCC and
                self.state[6, 4] != UNOCC and
                self.state[6, 5] != self.state[6, 4]):
                self.state[6, 5] = UNOCC

            if (self.state[5, 6] != UNOCC and
                self.state[4, 6] != UNOCC and
                self.state[5, 6] != self.state[4, 6]):
                self.state[5, 6] = UNOCC

            if (self.state[2, 6] != UNOCC and
                self.state[3, 6] != UNOCC and
                self.state[2, 6] != self.state[3, 6]):
                self.state[2, 6] = UNOCC

        if shrink_no == 2:
            if (self.state[2, 3] != UNOCC and
                self.state[2, 4] != UNOCC and
                self.state[2, 3] != self.state[2, 4]):
                self.state[2, 3] = UNOCC

            if (self.state[3, 2] != UNOCC and
                self.state[4, 2] != UNOCC and
                self.state[3, 2] != self.state[4, 2]):
                self.state[3, 2] = UNOCC

            if (self.state[5, 3] != UNOCC and
                self.state[5, 4] != UNOCC and
                self.state[5, 3] != self.state[5, 4]):
                self.state[5, 3] = UNOCC

            if (self.state[4, 5] != UNOCC and
                self.state[3, 5] != UNOCC and
                self.state[4, 5] != self.state[3, 5]):
                self.state[4, 5] = UNOCC

###############################################################################

    def update_board_inplace(self, action, colour):
        '''
        updates the internal board representation in-place
        '''
        action_tuple = np.array(action)
        action_size = action_tuple.size

        if action_size == 1:  # action is None
            return

        elif action_size == 2:  # placing phase
          self.state[action_tuple[0], action_tuple[1]] = colour

        elif action_size == 4:  # moving phase
          self.state[action_tuple[0][0], action_tuple[0][1]] = UNOCC
          self.state[action_tuple[1][0], action_tuple[1][1]] = colour

        self.eliminate_board(self.state, colour)
        self.move = action # update the move that brought it to this state

###############################################################################

    def pvs_estim(self):
        '''
        principal variation estimation function (see comments.txt)
        '''
        results = np.bincount(self.state.ravel())

        return (results[self.colour] - results[MAP[self.colour]])

###############################################################################

    def count_legal_moves(self, board, row, col, piece):
        '''
        count number of safe + legal moves available for 'piece'
        '''
        legal_moves = 0
        check_dir = {'D':[row+1 < SIZE, row+2 < SIZE],
                     'U':[row-1 >= 0, row-2 >= 0],
                     'R':[col+1 < SIZE, col+2 < SIZE],
                     'L':[col-1 >= 0, col-2 >= 0]}

        for m in check_dir:
            if check_dir[m][0]:
                row2, col2 = self.pos_check(board, row, col, m,
                                            return_rowcol=True)
                new_pos = self.state[row2, col2]

                if new_pos == UNOCC and not self.is_eliminated(board, row2,
                                                               col2, piece):
                    legal_moves += 1

                if new_pos == WHITE or new_pos == BLACK:
                    # check for jumps
                    if check_dir[m][1]:
                        row3, col3 = self.pos_check(board, row, col, '2' + m,
                                                    return_rowcol=True)
                        if (self.state[row3,col3] == UNOCC and not
                            self.is_eliminated(board, row3, col3, piece)):
                            legal_moves += 1

        return legal_moves

###############################################################################
# here
    def eval_func(self, phase):
        '''
        evaluation function for minimax (see comments.txt)
        '''
        opp_colour = MAP[self.colour]
        results = np.bincount(self.state.ravel())

        f1 = results[self.colour]  # player pieces
        f2 = results[opp_colour]  # enemy pieces
        f3 = 0  # count internal position score
        f4 = 0  # safe mobility
        f5 = 0  # connectedness

        if phase == 1:  # placing
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
                                if (self.pos_check(self.state, row, col, m) ==
                                    self.colour):
                                    f5 += 1


        if phase == 2:  #all moving phases
            if results[self.colour] < 2 and results[opp_colour] >= 2:
                return LOSE
            if results[self.colour] >= 2 and results[opp_colour] < 2:
                return WIN
            if results[self.colour] < 2 and results[opp_colour] < 2:
                return TIE

            for row, line in enumerate(self.state):
                for col, symbol in enumerate(line):
                    if symbol == self.colour:
                        f3 += PLACEMAP[self.colour][row][col]
                        f4 += self.count_legal_moves(self.state, row , col,
                                                     self.colour)
                        check_dir = {'D':row+1 < SIZE,
                                     'U':row-1 >= 0,
                                     'R':col+1 < SIZE,
                                     'L':col-1 >= 0}

                        for m in check_dir:
                            if check_dir[m]:
                                if (self.pos_check(self.state, row, col, m) ==
                                    self.colour):
                                    f5 += 1

        return (f1*WEIGHTS[0] + f2*WEIGHTS[1] + f3*WEIGHTS[2] + f4*WEIGHTS[3]
                + f5*WEIGHTS[4])

###############################################################################

    def gen_child(self, colour):
        '''
        generate child nodes of a state (from parta)
        '''
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
                            to_check = self.pos_check(self.state,row,col,dir)

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
                                    if (self.pos_check(self.state,row,col,j)
                                        == UNOCC):
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
        '''
        generate child nodes
        '''
        child_nodes = []

        for row, line in enumerate(self.state):
            for col, symbol in enumerate(line):
                if row not in DEATHMAP[colour]:
                    if self.state[row, col] == UNOCC:
                        (child_nodes.append(self.update_board_return(
                            (row, col), colour)))

        return child_nodes

###############################################################################

    def gen_child_place_agr(self, colour):
        '''
        aggressively prunes by only expanding nodes in a particular order
            (placing)
        '''
        child_nodes = []
        board_val = ORDERMAP[colour]

        for placements in board_val:
            if placements[0] not in DEATHMAP[colour]:
                if self.state[placements[0], placements[1]] == UNOCC:
                    (child_nodes.append(self.update_board_return(
                        (placements[0], placements[1]), colour)))

        return child_nodes

###############################################################################
