# Chirag Rao Sahib      : 836011
# Daniel Hao            : 834496
# Date                  : 28/03/2018
# Python version        : 3.6.4

###############################################################################

import copy
from collections import defaultdict

NODES_GENERATED = 2e10
MAX_DEPTH = 2e10
DEADEND = 1e5
BLACK_MULTIPLIER = 1e3
BLACK = '@'
WHITE = 'O'
UNOCC = '-'
CORNER = 'X'
SIZE = 8
MODS = {'R': (0, 1),
        '2R': (0, 2),
        'L': (0, -1),
        '2L': (0, -2),
        'D': (1, 0),
        '2D': (2, 0),
        'U': (-1, 0),
        '2U': (-2, 0)}


###############################################################################

def posCheck(board, row, col, dir):
    '''
    returns symbol at a given board position (modified by direction)
    '''
    return board[row + MODS[dir][0]][col + MODS[dir][1]]

###############################################################################

def checkSurround(board, row, col):
    '''
    given a position counts possible movements
    '''
    availMoves = 0
    checkCond = {'D':[row+1 < SIZE, row+2 < SIZE],
                 'U':[row-1 >= 0, row-2 >= 0],
                 'R':[col+1 < SIZE, col+2 < SIZE],
                 'L':[col-1 >= 0, col-2 >= 0]}

    for direction in checkCond:
        if checkCond[direction][0]:
            symbol = posCheck(board, row, col, direction)
            if symbol == UNOCC:
                availMoves += 1
            if symbol == WHITE or symbol == BLACK:
                # check whether jump is possible
                if checkCond[direction][1]:
                    if posCheck(board, row, col,'2' + direction) == UNOCC:
                        availMoves += 1

    return availMoves

###############################################################################

def testMoves(gameState):
    '''
    iterates over board and counts available moves for each piece
    '''
    whiteMoves = 0
    blackMoves = 0

    for row, line in enumerate(gameState):
        for col, symbol in enumerate(line):
            if symbol == WHITE:
                whiteMoves += checkSurround(gameState, row, col)
            if symbol == BLACK:
                blackMoves += checkSurround(gameState, row, col)

    return whiteMoves, blackMoves

###############################################################################

class game():

    def __init__(self, initialState):
        self.initialState = initialState

###############################################################################

    def movePiece(self, state, posA, posB, piece = WHITE):
        '''
        returns updated board after moving a piece from A to B
        '''
        newBoard = copy.deepcopy(state)
        newBoard[int(posA[0])][int(posA[1])] = UNOCC
        newBoard[int(posB[0])][int(posB[1])] = piece
        newBoard = self.updateBoard(newBoard)

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

    def getAvailableMoves(self, state):
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

                if symbol == WHITE:
                    for dir in checkCond:
                        if checkCond[dir][0]:
                            posToCheck = posCheck(state, row, col, dir)
                            index = str(row) + str(col)

                            if posToCheck == UNOCC:
                                tmpA = row + checkCond[dir][1]
                                tmpB = col + checkCond[dir][2]
                                tmpIndex = str(tmpA) + str(tmpB)
                                actions[index].append(tmpIndex)

                            elif posToCheck == WHITE or posToCheck == BLACK:
                                # check whether jump is possible
                                if checkCond[dir][3]:
                                    if posCheck(state, row, col,'2' + dir) == UNOCC:
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
            check = [posCheck(board, row, col, i) for i in ['L','R','U','D']]
            if check[0] == flag or check[0] == CORNER:
                if check[1] == flag or check[1] == CORNER:
                    return True
            if check[2] == flag or check[2] == CORNER:
                if check[3] == flag or check[3] == CORNER:
                    return True

        return False

###############################################################################

    def eliminate(self, state, row, col):
        '''
        returns updated board after eliminating a given piece
        '''
        newBoard = copy.deepcopy(state)
        newBoard[row][col] = UNOCC

        return newBoard

###############################################################################

    def updateBoard(self, state):
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
                            newBoard = self.eliminate(newBoard, row, col)

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

    def aStarSearch(self, max_nodes, maxDepth):
        # Performs a-star search
        # Prints the list of solution moves and the solution length

        # Need a dictionary for the frontier and for the expanded nodes
        frontier_nodes = {}  # about to expand nodes
        expanded_nodes = {}

        self.starting_state = copy.deepcopy(self.initialState)
        current_state = copy.deepcopy(self.initialState)

        # Node index is used for indexing the dictionaries and to keep track of the number of nodes expanded
        node_index = 0

        # Set the first element in both dictionaries to the starting state
        # This is the only node that will be in both dictionaries
        expanded_nodes[node_index] = {"state": current_state, "parent": "root", "action": "start",
                                   "total_cost": self.evalFunc(current_state, 0), "depth": 0}

        frontier_nodes[node_index] = {"state": current_state, "parent": "root", "action": "start",
                                   "total_cost": self.evalFunc(current_state, 0), "depth": 0}


        isSolution = True

        # all_nodes keeps track of all nodes on the frontier and is the priority queue.
        # Each element in the list is a tuple consisting of node index and total cost of the node.
        all_frontier_nodes = [(0, frontier_nodes[0]["total_cost"])]

        # Stop when maximum nodes or depth have been considered
        while isSolution:

            # Get current depth of state for use in total cost calculation
            current_depth = 0

            for node_num, node in expanded_nodes.items():
                if node["state"] == current_state:
                    current_depth = node["depth"]

            # Find available actions for the current state
            available_actions = self.getAvailableMoves(current_state)


            # Iterate through possible actions
            for start, value in available_actions.items():
                for end in value:
                    visited = False


                    # If max nodes reached stop searching
                    if node_index >= max_nodes:
                        print("No Solution Found in first {} nodes generated".format(max_nodes))
                        isSolution = False

                    # if max depth reached stop searching
                    if current_depth >= maxDepth:
                        print("No Solution Found in first {} layers".format(maxDepth))
                        isSolution = False


                    # Find the new state corresponding to the action and calculate total cost
                    if self.isValidMove(current_state, start, end):
                        new_state = self.movePiece(current_state, start, end)
                    else:
                        continue

                    new_state_parent = copy.deepcopy(current_state)

                    # Check to see if new state has already been expanded
                    for expanded_node in expanded_nodes.values():
                        if expanded_node["state"] == new_state:
                            if expanded_node["parent"] == new_state_parent:
                                visited = True

                    # Check to see if new state and parent is on the frontier
                    # The same state can be added twice to the frontier if the parent state is different
                    for frontier_node in frontier_nodes.values():
                        if frontier_node["state"] == new_state:
                            if frontier_node["parent"] == new_state_parent:
                                visited = True

                    # If new state has already been expanded or is on the frontier, continue with next action
                    if visited:
                        continue

                    else:
                        # Each action represents another node generated
                        node_index += 1
                        depth = current_depth + 1

                        # Total cost is path length (number of steps from starting state) + heuristic
                        new_state_cost = self.evalFunc(new_state, depth)


                        # Add the node index and total cost to the all_nodes list
                        all_frontier_nodes.append((node_index, new_state_cost))

                        # Add the node to the frontier
                        frontier_nodes[node_index] = {"state": new_state, "parent": new_state_parent,  "total_cost": new_state_cost, "depth": current_depth + 1, "action" : '({}, {}) -> ({}, {})'.format(start[1], start[0], end[1], end[0])}

            # Sort all the nodes on the frontier by total cost
            all_frontier_nodes = sorted(all_frontier_nodes, key=lambda x: x[1])

            # If the number of nodes generated does not exceed max nodes, find the best node and set the current state to that state
            if isSolution:

                # The best node will be at the front of the queue
                # After selecting the node for expansion, remove it from the queue
                best_node = all_frontier_nodes.pop(0)
                best_node_index = best_node[0]
                best_node_state = frontier_nodes[best_node_index]["state"]
                current_state = best_node_state

                # Move the node from the frontier to the expanded nodes
                expanded_nodes[best_node_index] = (frontier_nodes.pop(best_node_index))

                # Check if current state is goal state
                if self.isComplete(best_node_state):
                    for node_num, node in expanded_nodes.items():
                        if self.isComplete(node["state"]):
                            final_node = expanded_nodes[node_num]
# =============================================================================
                    # print(final_node)
                    # for a in final_node['state']:
                    #     print(a)
                    # print('ontop')
                    # for a in final_node['parent']:
                    #     print(a)
# =============================================================================
                    finalresult = self.generate_solution_path(final_node, expanded_nodes, [])
                    # Display the solution path
                    for a in reversed(finalresult):
                        print(a)
                    # +1 the depth
                    print(current_depth, node_index)
                    break

###############################################################################

    def generate_solution_path(self, node, node_dict, result):
        # Return the solution path for display from final (goal) state to starting state
        # If the node is the root, return the path
        if node["parent"] == "root":
            # If root is found, add the node and then return
            return result

        else:
            # If the node is not the root, add the state and action to the solution path
            parent_state = node["parent"]
            result.append(node['action'])

            # Find the parent of the node and recurse
            for node_num, expanded_node in node_dict.items():
                if expanded_node["state"] == parent_state:
                    return self.generate_solution_path(expanded_node, node_dict, result)


###############################################################################

def main():
    gameState = []

    for i in range(SIZE):
        gameState.append(input().split())

    mode = input().split()[0]

    if mode == 'Moves':
        whiteMoves, blackMoves = testMoves(gameState)
        print('{}\n{}'.format(whiteMoves, blackMoves))
    elif mode == 'Massacre':
        board = game(gameState)
        board.aStarSearch(NODES_GENERATED, MAX_DEPTH)
    else:
        print('Invalid mode')

###############################################################################

main()


