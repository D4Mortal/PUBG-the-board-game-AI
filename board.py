from util import UNOCC, BLACK, WHITE, CORNER, MAP, WEIGHTS, PLACEMAP, \
SIZE, LOSE, TIE, WIN, CHECK_ORDER_WHITE, CHECK_ORDER_BLACK, DEATHMAP

from util import pos_check

import numpy as np

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
        f1 = results[self.colour] - results[oppColour]
        f2 = 0
        f3 = 0
        f4 = 0  # connectdness

        if phase == 1:  #placing
            # center control + connectedness
            for row, line in enumerate(self.state):
                for col, symbol in enumerate(line):
                    if symbol  == self.colour:
                        f2 += PLACEMAP[self.colour][row][col][row][col]
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

    # This aggressively prunes by only only expanding nodes in a particular 
    # order
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