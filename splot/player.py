from util import UNOCC, BLACK, WHITE, CORNER, MAP, SIZE, LOSE, TIE, WIN, WALL,\
PHASE1, IDEAL_DEPTH, ZOR

from util import zorHash, hashRemove, hashMove

from board import board
import time
import numpy as np


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
            # print("i hope this is proc")
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
            print(total_branching)
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
                # print(self.node.state)
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
                    if depth == self.abHash[nodeHash][1]:
                        alpha, beta = self.abHash[nodeHash][0]

            else:
                nodeValue = node.eval_func(2)
                self.hashTable[nodeHash] = nodeValue
                if alpha != -np.inf and beta != np.inf:
                    self.abHash[nodeHash] = ((alpha, beta), depth)

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
                    if depth == self.abHash[nodeHash][1]:
                        alpha, beta = self.abHash[nodeHash][0]

            else:
                nodeValue = node.eval_func(2)
                self.hashTable[nodeHash] = nodeValue
                if alpha != -np.inf and beta != np.inf:
                    self.abHash[nodeHash] = ((alpha, beta), depth)

            if  depth <= 0 or nodeValue == LOSE or nodeValue == WIN or nodeValue == TIE:
                return nodeValue

            v = np.inf
            ordered_child_nodes = sorted(node.genChild(MAP[node.colour]),
                key=lambda x: x[0].move_estim)

            for child in ordered_child_nodes:

                v = min(v, maxValue(child, depth-1, alpha, beta, turns+1, nodeHash))
#                print(child.eval_node(), end='')
#                print(child.state)
                if v <= alpha: return v
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
    
    
    
    
def testrun(me = 'white'):
     game = Player(me)


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



     print("This is the current board config")
     print(game.node.state)
     depth = input("Please select a depth to search on: ")
     print("Searching ahead for {} moves...".format(depth))
     result = game.miniMax(int(depth), game.node.genChild(game.player_colour))
     print("The optimal move for white is: ", end='')
     print(result)

testrun()