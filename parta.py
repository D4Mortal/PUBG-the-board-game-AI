# Chirag Rao Sahib      : 836011
# Daniel Hao            : 834496
# Date                  : 30/03/2018
# Python version        : 3.6.4

###########################s####################################################

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

class game():

    def __init__(self, initialState):
        # eliminate pieces in initial state if possible
        self.initialState = self.eliminateBoard(initialState)

###############################################################################

    def posCheck(self, state, row, col, dir):
        '''
        returns symbol at a given board position (modified by direction)
        '''
        return state[row + MODS[dir][0]][col + MODS[dir][1]]

###############################################################################

    def moveCount(self, piece):
        '''
        counts available moves for a given piece
        '''
        moves = 0

        for v in self.getLegalMoves(self.initialState, piece).values():
            moves += len(v)

        return(moves)

###############################################################################

    def movePiece(self, state, posA, posB, piece = WHITE):
        '''
        returns updated board after moving a piece from A to B
        '''
        newBoard = copy.deepcopy(state)
        newBoard[int(posA[0])][int(posA[1])] = UNOCC
        newBoard[int(posB[0])][int(posB[1])] = piece
        newBoard = self.eliminateBoard(newBoard)

        return newBoard

###############################################################################

    def isValidMove(self, state, posA, posB):
        '''
        verify that a given move (A to B) is valid
        '''
        rowA = int(posA[0])
        colA = int(posA[1])
        rowB = int(posB[0])
        colB = int(posB[1])

        if rowB > 7 or rowB < 0 or colB > 7 or colB < 0:
            # check move is within board
            return False

        if rowA != rowB and colA != colB:
            # check for diagonal movement
            return False

        if state[rowB][colB] != UNOCC:
            # check whether position is occupied
            return False

        return True

###############################################################################

    def isComplete(self, state):
        '''
        check whether a given state is a goal state (complete)
        '''
        for row in state:
            for symbol in row:
                if symbol == BLACK:
                    return False

        return True

###############################################################################

    def heuristic(self, state):
        '''
        heuristic function that returns the 'score' of a state.
        lower scores indicate we are closer to a goal, while higher
        scores indicate we are further away
        '''
        if self.isComplete(state):
            return 1

        stateScore = 0
        blackCount = 0
        whiteCount = 0

        # count number of each piece on board
        for row in state:
            for symbol in row:
                if symbol == BLACK:
                    blackCount += 1
                if symbol == WHITE:
                    whiteCount += 1

        if whiteCount < 1:
            # if no white pieces left state is a deadend (increase score)
            stateScore += DEADEND

        # increase score in proportion to number of black pieces
        # since more black pieces on board is less favourable
        stateScore += blackCount * BLACK_MULTIPLIER

        # also consider manhattan distance
        stateScore += self.totalManHatDist(state)

        return stateScore

###############################################################################

    def evalFunc(self, state, currentDepth):
        '''
        evaluation function as per f(n) = h(n) + g(n)
        '''
        return self.heuristic(state) + currentDepth

###############################################################################

    def getLegalMoves(self, state, piece = WHITE):
        '''
        return dictionary of moves available for each white piece
        '''
        actions = defaultdict(list)

        for row, line in enumerate(state):
            # describe up/down moves to check
            checkCond = {'D': [row+1 < SIZE, 1, 0, row+2 < SIZE, 2, 0],
                         'U': [row-1 >= 0, -1, 0, row-2 >= 0, -2, 0]}

            for col, symbol in enumerate(line):
                # describe left/right moves to check
                checkCond['R'] = [col+1 < SIZE, 0, 1, col+2 < SIZE, 0, 2]
                checkCond['L'] = [col-1 >= 0, 0, -1, col-2 >= 0, 0, -2]

                if symbol == piece:
                    for dir in checkCond:
                        if checkCond[dir][0]:
                            posToCheck = self.posCheck(state, row, col, dir)
                            index = str(row) + str(col)

                            if posToCheck == UNOCC:
                                tmpA = row + checkCond[dir][1]
                                tmpB = col + checkCond[dir][2]
                                tmpIndex = str(tmpA) + str(tmpB)
                                actions[index].append(tmpIndex)

                            elif posToCheck == WHITE or posToCheck == BLACK:
                                # check whether jump is possible
                                if checkCond[dir][3]:
                                    j = '2' + dir  # jump direction
                                    if self.posCheck(state,row,col,j) == UNOCC:
                                        tmpA = row + checkCond[dir][4]
                                        tmpB = col + checkCond[dir][5]
                                        tmpIndex = str(tmpA) + str(tmpB)
                                        actions[index].append(tmpIndex)

        return actions

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
            checkLeft = self.posCheck(board, row, col, 'L')
            checkRight = self.posCheck(board, row, col, 'R')
            if checkLeft == flag or checkLeft == CORNER:
                if checkRight == flag or checkRight == CORNER:
                    return True

        elif col == 0 or col == 7:
            checkUp = self.posCheck(board, row, col, 'U')
            checkDown = self.posCheck(board, row, col, 'D')
            if checkUp == flag or checkUp == CORNER:
                if checkDown == flag or checkDown == CORNER:
                    return True

        else:
            # generate positions to check
            check = [self.posCheck(board,row,col,i) for i in ['L','R','U','D']]
            if check[0] == flag or check[0] == CORNER:
                if check[1] == flag or check[1] == CORNER:
                    return True
            if check[2] == flag or check[2] == CORNER:
                if check[3] == flag or check[3] == CORNER:
                    return True

        return False

###############################################################################

    def eliminateBoard(self, state):
        '''
        returns updated board after necessary eliminations
        '''
        newBoard = copy.deepcopy(state)

        # white has elimination priority, try eliminate black pieces first
        for piece in [BLACK, WHITE]:
            for row, line in enumerate(newBoard):
                for col, symbol in enumerate(line):
                    if symbol == piece:
                        if self.isEliminated(newBoard, row, col, piece):
                            newBoard[row][col] = UNOCC

        return newBoard

###############################################################################

    def manHatDist(self, posA, posB):
        '''
        calculate the Manhattan distance between two pieces (A and B).
        the sum of the row and column differences for A and B
        '''
        dist = abs(int(posA[0]) - int(posB[0]))
        dist += abs(int(posA[1]) - int(posB[1]))

        return dist

###############################################################################

    def totalManHatDist(self, state):
        '''
        total Manhattan distance from every white piece to every black piece
        '''
        total = 0

        for row, line in enumerate(state):
            for col, symbol in enumerate(line):
                if symbol == BLACK:
                    for row2, line2 in enumerate(state):
                        for col2, symbol2 in enumerate(line2):
                            if symbol2 == WHITE:
                                posA = str(row) + str(col)
                                posB = str(row2) + str(col2)
                                total += self.manHatDist(posA, posB)

        return total

###############################################################################

    def aStarSearch(self):
        '''
        returns the shortest path to a goal state
        '''
        frontierNodes = {}
        expandedNodes = {}
        currentState = copy.deepcopy(self.initialState)
        nodeIndex = 0  # track no. nodes expanded + used to index dictionaries

        # current state is part of both expanded and frontier nodes
        expandedNodes[nodeIndex] = {'state': currentState,
                                     'parent': 'root',
                                     'action': 'start',
                                     'totalCost': self.evalFunc(currentState,
                                        0),
                                     'depth': 0}
        frontierNodes[nodeIndex] = copy.deepcopy(expandedNodes[nodeIndex])
        limitReached = False  # terminal condition for loop
        # priority queue of frontier nodes
        # each element is a tuple : (node index, total cost of a node)
        priorityQueue = [(0, frontierNodes[0]['totalCost'])]

        while limitReached == False:
            depth = 0  # track depth of solution

            for n in expandedNodes.values():
                # get depth of current state
                if n['state'] == currentState:
                    depth = n['depth']
                    break

            # find legal moves in the current state
            legalMoves = self.getLegalMoves(currentState)

            # iterate through legal moves
            for posA, value in legalMoves.items():
                for posB in value:
                    visited = False

                    # stop searching if node or depth limit reached
                    if nodeIndex >= MAX_NODES or depth >= MAX_DEPTH:
                        print('Node or depth limit reached')
                        limitReached = True
                        break

                    # generate test state for a move
                    if self.isValidMove(currentState, posA, posB):
                        tState = self.movePiece(currentState, posA, posB)
                    else:
                        # skip and test validness of next move (state)
                        continue

                    tStateParent = copy.deepcopy(currentState)

                    # Check if new state is already expanded
                    for n in expandedNodes.values():
                        if n['state'] == tState:
                            if n['parent'] == tStateParent:
                                visited = True
                                break

                    # Check if new state is in frontier nodes
                    # note: state can be in frontier twice if
                    #       parent state is different
                    for fn in frontierNodes.values():
                        if fn['state'] == tState:
                            if fn['parent'] == tStateParent:
                                visited = True
                                break

                    if visited:
                        # skip state if expanded or in frontier
                        continue
                    else:
                        # each move represents another node generated
                        nodeIndex += 1
                        nodeDepth = depth + 1

                        # total cost as per evaluation function f(n)
                        tStateCost = self.evalFunc(tState, nodeDepth)
                        priorityQueue.append((nodeIndex, tStateCost))

                        # Add node to frontier
                        frontierNodes[nodeIndex] = {'state': tState,
                                                    'parent': tStateParent,
                                                    'totalCost': tStateCost,
                                                    'depth': nodeDepth,
                                                    'action' :
                                                    '({}, {}) -> ({}, {})'
                                                        .format(posA[1],
                                                            posA[0], posB[1],
                                                            posB[0])}

            # sort frontier nodes by total cost f(n)
            priorityQueue = sorted(priorityQueue, key = lambda x: x[1])

            if limitReached == False:
                bestNode = priorityQueue.pop(0)  # best node at front of queue
                bestNodeIndex = bestNode[0]

                # move best node from frontier to expanded nodes
                bestNodeState = frontierNodes[bestNodeIndex]['state']
                expandedNodes[bestNodeIndex] = frontierNodes.pop(bestNodeIndex)
                currentState = bestNodeState  # update current state

                # test if current state is now goal state
                if self.isComplete(currentState):
                    for index, node in expandedNodes.items():
                        if self.isComplete(node['state']):
                            # find a goal node
                            goalNode = expandedNodes[index]
                            break

                    # generate solution path if complete and goal node found
                    path = self.genSolPath(goalNode, expandedNodes, [])

                    return reversed(path)  # get path from root to goal node

###############################################################################

    def genSolPath(self, node, nodeDict, path):
        '''
        recursive traversal up the search tree from the goal state to root,
            reconstructing the solution path
        '''
        parentState = node['parent']

        if parentState == 'root':
            # base case : if the node is the root, return the path
            return path
        else:
            path.append(node['action'])
            # traverse up the node's parent
            for n in nodeDict.values():
                if n['state'] == parentState:
                    return self.genSolPath(n, nodeDict, path)


###############################################################################
                    
def printBoard(board):
    for i in board:
        for a in i:
            print(a + " ",end = '')
        print('')
        
###############################################################################       
        
def main():
    gameState = []
    
    for i in range(SIZE):
        gameState.append(input().split())

    mode = input().split()[0]
    board = game(gameState)  # initialise game class

    if mode == 'Moves':
        print("The board config is:")
        printBoard(gameState)
        print('{}\n{}'.format(board.moveCount(WHITE), board.moveCount(BLACK)))
    elif mode == 'Massacre':
        # Display the solution path
        print("The board config is:")
        printBoard(gameState)
        for move in board.aStarSearch():
            print(move)
    else:
        print('Invalid mode')

###############################################################################

main()
