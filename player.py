# Chirag Rao Sahib      : 836011
# Daniel Hao            : 834496
# Date                  : 10/05/2018
# Python version        : 3.6.4

# Module                : player.py

###############################################################################

import copy
import numpy as np
from zobrist_hash import *
from constants import *
from board import *

###############################################################################

class Player():

    def __init__(self, colour):
        # initialise internal board representation
        self.state = np.full((SIZE, SIZE), UNOCC, dtype=int)
        self.state[0,0] = CORNER
        self.state[0,7] = CORNER
        self.state[7,0] = CORNER
        self.state[7,7] = CORNER
        self.turns = 0  # for current phase
        self.total_turns = 0  # for entire game
        self.hash_table = dict()  # board evaluation for a given hash
        self.ab_hash = dict()  # alpha beta value from minimax
        self.zob = init_table()  # initialise zobrist hash table

        if colour[0] == 'w':
          self.player_colour = WHITE
          self.opp_colour = BLACK
          # create root node for minimax search tree
          self.node = board(self.state, None, WHITE)

        else:
          self.player_colour = BLACK
          self.opp_colour = WHITE
          self.node = board(self.state, None, BLACK)

###############################################################################

    def action(self, turns):
        '''
        returns an action as per specs
        '''
        if self.player_colour == WHITE:
            if turns == PHASE2-1:
                self.shrink_board(self.node, 1)
                self.node.shrink_eliminate(1)

            if turns == PHASE3-1:
                self.shrink_board(self.node, 2)
                self.node.shrink_eliminate(2)

        self.turns = turns  # only used by player pieces

        if self.total_turns > PHASE1:
            # generate child nodes for player
            child_nodes_friendly = self.node.gen_child(self.player_colour)

            # calculate total branching factor (see comments.txt)
            total_branching = (len(child_nodes_friendly) +
                               len(self.node.gen_child(self.opp_colour)))

            # decide action via minimax
            action = self.mini_max(D_DEPTH[total_branching],
                                   child_nodes_friendly)

            self.total_turns += 1
            self.node.update_board_inplace(action, self.player_colour)

            if action == None:  # return None if no action possible
                return None

            # inverted as per (col, row)
            return (action[0][::-1], action[1][::-1])

        else:
            # minimax for placing phase
            self.total_turns += 1
            place_move = self.mini_max_place(PLACE_DEPTH)
            self.node.update_board_inplace(place_move, self.player_colour)

            return place_move[::-1]

###############################################################################

    def update(self, action):
        '''
        update board from opponent action
        '''
        size = np.array(action).size

        if self.player_colour == BLACK:
            if self.turns == PHASE2-2:
                self.shrink_board(self.node, 1)
                self.node.shrink_eliminate(1)

            if self.turns == PHASE3-2:
                self.shrink_board(self.node, 2)
                self.node.shrink_eliminate(2)

        if size == 2:  # placing phase
            self.node.update_board_inplace(action[::-1], self.opp_colour)

        if size == 4:  # moving phase
            invert1 = action[0][::-1]
            invert2 = action[1][::-1]
            self.node.update_board_inplace((invert1, invert2), self.opp_colour)

        self.total_turns += 1

###############################################################################

    def shrink_board(self, node, shrink_no):
        '''
        shrink the board as per specs
        '''
        if shrink_no == 1:  # first shrink
            node.state[0, :] = WALL
            node.state[7, :] = WALL
            node.state[:, 0] = WALL
            node.state[:, 7] = WALL
            node.state[1,1] = CORNER
            node.state[1,6] = CORNER
            node.state[6,1] = CORNER
            node.state[6,6] = CORNER

        if shrink_no == 2:  # second shrink
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
        '''
        minimax for moving phase (adapted from AIMA)
        '''
        # hash current state (zobrist)
        current_hash = zor_hash(self.zob, self.node.state)

        def max_val(node_info, depth, alpha, beta, turns, hash_val):
            node = node_info[0]  # node itself
            killed = node_info[1]  # eliminated pieces

            if turns == PHASE2:
                # first shrinking + elimination
                self.shrink_board(node, 1)
                node.shrink_eliminate(1)
                # hash shrinked state
                node_hash = zor_hash(self.zob, node.state)

            elif turns == PHASE3:
                self.shrink_board(node, 2)
                node.shrink_eliminate(2)
                # hash shrinked state
                node_hash = zor_hash(self.zob, node.state)

            else:
                # update hash table with new move
                node_hash = hash_mv(self.zob, hash_val, self.opp_colour,
                                    node.move)

                # update hash table with eliminated pieces
                for dead in killed:
                    node_hash = hash_rm(self.zob, node_hash, dead[1], dead[0])

            # avoid recomputation of board evaluation value, performs lookup
            if node_hash in self.hash_table:
                node_val = self.hash_table[node_hash]

                # avoid recomputation of alpha-beta hash, performs lookup
                if node_hash in self.ab_hash:
                    if depth == self.ab_hash[node_hash][1]:
                        alpha, beta = self.ab_hash[node_hash][0]

            # otherwise compute and store the board eval value
            else:
                node_val = node.eval_func(2)
                self.hash_table[node_hash] = node_val

                # only bother storing alpha-beta hash if not inf
                if alpha != -np.inf and beta != np.inf:
                    self.ab_hash[node_hash] = ((alpha, beta), depth)

            # reached terminal state, or passed depth limit
            if depth <= 0 or node_val in {WIN, LOSE, TIE}:
                return node_val

            v = -np.inf

            # using pvs_estim find optimal move ordering (see comments.txt)
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

        # below is reversed logic of the above
        def min_val(node_info, depth, alpha, beta, turns, hash_val):
            node = node_info[0]
            killed = node_info[1]

            if turns == PHASE2:
                self.shrink_board(node, 1)
                node.shrink_eliminate(1)
                node_hash = zor_hash(self.zob, node.state)

            elif turns == PHASE3:
                self.shrink_board(node, 2)
                node.shrink_eliminate(2)
                node_hash = zor_hash(self.zob, node.state)

            else:
                node_hash = hash_mv(self.zob, hash_val, self.player_colour,
                                    node.move)
                for dead in killed:
                    node_hash = hash_rm(self.zob, node_hash, dead[1], dead[0])

            if node_hash in self.hash_table:
                node_val = self.hash_table[node_hash]

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
            # note: ordering in reverse manner to above
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

        # sorting child nodes per pvs_estim, 'best' states first
        child_nodes = sorted(child, key=lambda x: x[0].move_estim,
                                    reverse=True)

        for child in child_nodes:
            v = min_val(child, depth-1, best_score, beta, self.turns,
                        current_hash)
            if v > best_score:
                best_score = v
                best_action = child[0].move

        return best_action

###############################################################################

    def mini_max_place(self, depth):
        '''
        minimax for placing phase (same logic as mini_max())
        see comments.txt for aggressive pruning explanation
        '''
        def max_val(node_info, depth, alpha, beta):
            node = node_info[0]
            node_val = node.eval_func(1)

            if depth <= 0 or node_val in {WIN, LOSE, TIE}:
                return node_val

            v = -np.inf

            # move ordering to make pruning more efficient (see comments)
            # also ensures centre control
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

        for child in self.node.gen_child_place_agr(self.player_colour):
            v = min_val(child, depth-1, best_score, beta)

            if v > best_score:
                best_score = v
                best_action = child[0].move

        return best_action

###############################################################################
