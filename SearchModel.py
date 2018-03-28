from collections import defaultdict
import copy
BLACK = '@'
WHITE = 'O'
UNOCC = '-'
CORNER = 'X'
SIZE = 8
MODS = {'R':(0,1),
       '2R':(0,2),
       'L': (0,-1),
       '2L': (0,-2),
       'D':(1,0),
       '2D':(2,0),
       'U':(-1,0),
       '2U':(-2,0)}



def move(board, row, col, dir):
    '''
    '''
    return board[row + MODS[dir][0]][col + MODS[dir][1]]





class game():
    def __init__(self, initialState):
        self.initialState = initialState

    
    
    def makeMove(self, state, start, end, colour = WHITE):
        newBoard = copy.deepcopy(state)
        
        newBoard[int(start[0])][int(start[1])] = UNOCC
        newBoard[int(end[0])][int(end[1])] = colour
        
        newBoard = self.updateBoard(newBoard)
        return newBoard
    
    def isValidMove(self, state, start, end):
        row = int(start[0])
        col = int(start[1])
        rowEnd = int(end[0])
        colEnd = int(end[1])
        
        if rowEnd > 7 or rowEnd < 0 or colEnd > 7 or colEnd < 0:
            return False
        
        if row != rowEnd and col != colEnd:
            return False
        
        if state[rowEnd][colEnd] != UNOCC:
            return False
        
        return True
        
        
        
    def isComplete(self, state):
        for row in state:
            for element in row:
                if element == BLACK:
                    return False
        return True
    
    
    def heuristics(self, state):
        if self.isComplete(state):
            return 1
        
        BlackCount = 0
        WhiteCount = 0
        score = 0
        for row in state:
            for element in row:
                if element == BLACK: BlackCount += 1
                if element == WHITE: WhiteCount += 1
        if WhiteCount < 2:
            score += 999999999999999
            
        score += BlackCount * 1000
        
        score += self.findTotalDistance(state) 
        
        return score
    
    
    def findTotalCost(self, state, depth):
        return self.heuristics(state) + depth
    
    
    # find all the available moves on a board regardless of if the white piece will die making the move.
    def getAvailableMoves(self, state):
        
        row = 0
        actions = defaultdict(list)
        
        for r in state:
            col = 0
            

            
            for element in r:
                if element == WHITE:

                    if row + 1 < 8:
                        if move(state, row, col, 'D') == UNOCC:
                            actions[str(row) + str(col)].append(str(row + 1) + str(col))
        
                        elif move(state, row, col, 'D') == WHITE or move(state, row, col, 'D') == BLACK:
                            if row + 2 < 8:
                                if move(state, row, col, '2D') == UNOCC:
                                     actions[str(row) + str(col)].append(str(row + 2) + str(col))
    
                    if row - 1 >= 0:
                        if move(state, row, col, 'U') == UNOCC:
                            actions[str(row) + str(col)].append(str(row - 1) + str(col))

        
                        elif move(state, row, col, 'U') == WHITE or move(state, row, col, 'U') == BLACK:
                            if row - 2 >= 0:
                                if move(state, row, col, '2U') == UNOCC:
                                    actions[str(row) + str(col)].append(str(row - 2) + str(col))
            
                    
                    
                    if col + 1 < 8:
                        if move(state, row, col, 'R') == UNOCC:
                            actions[str(row) + str(col)].append(str(row) + str(col + 1))
        
                        elif move(state, row, col, 'R') == WHITE or move(state, row, col, 'R') == BLACK:
                            if col + 2 < 8:
                                if move(state, row, col, '2R') == UNOCC:
                                    actions[str(row) + str(col)].append(str(row) + str(col + 2))
                    
                    if col - 1 >= 0:
                        if move(state, row, col, 'L') == UNOCC:
                            actions[str(row) + str(col)].append(str(row) + str(col - 1))
        
                        elif move(state, row, col, 'L') == WHITE or move(state, row, col, 'L') == BLACK:
                            if col - 2 >= 0:
                                if move(state, row, col, '2L') == UNOCC:
                                    actions[str(row) + str(col)].append(str(row) + str(col - 2))
    
                col += 1
            row += 1
            
        return actions
        
    
    def isDead(self, board, row, col, colour):
        
        if colour == WHITE:
            
            if row == 0 or row == 7:
                if move(board, row, col, "L") == BLACK or  move(board, row, col, "L") == CORNER:
                    if move(board, row, col, "R") == BLACK or  move(board, row, col, "R") == CORNER:
                        return True
            elif col == 0 or col == 7:
                if move(board, row, col, "U") == BLACK or  move(board, row, col, "U") == CORNER:
                    if move(board, row, col, "D") == BLACK or  move(board, row, col, "D") == CORNER:
                        return True
                    
            else:
                if move(board, row, col, "L") == BLACK or  move(board, row, col, "L") == CORNER:
                    if move(board, row, col, "R") == BLACK or  move(board, row, col, "R") == CORNER:
                        return True
                if move(board, row, col, "U") == BLACK or  move(board, row, col, "U") == CORNER:
                    if move(board, row, col, "D") == BLACK or  move(board, row, col, "D") == CORNER:
                        return True
        
        if colour == BLACK:
            
            if row == 0 or row == 7:
                if move(board, row, col, "L") == WHITE or  move(board, row, col, "L") == CORNER:
                    if move(board, row, col, "R") == WHITE or  move(board, row, col, "R") == CORNER:
                        return True
            elif col == 0 or col == 7:
                if move(board, row, col, "U") == WHITE or  move(board, row, col, "U") == CORNER:
                    if move(board, row, col, "D") == WHITE or  move(board, row, col, "D") == CORNER:
                        return True
                    
            else:
                if move(board, row, col, "L") == WHITE or  move(board, row, col, "L") == CORNER:
                    if move(board, row, col, "R") == WHITE or  move(board, row, col, "R") == CORNER:
                        return True
                if move(board, row, col, "U") == WHITE or  move(board, row, col, "U") == CORNER:
                    if move(board, row, col, "D") == WHITE or  move(board, row, col, "D") == CORNER:
                        return True
        
        return False
    
    
    def remove(self, state, row, col):
        newBoard = copy.deepcopy(state)
        newBoard[row][col] = UNOCC
        return newBoard 
    
    
    def updateBoard(self, state):
        newBoard = copy.deepcopy(state)
        row = 0
        for r in newBoard:
            col = 0
            for element in r:
                if element == BLACK: 
                    if self.isDead(newBoard, row, col, BLACK):
                        newBoard = self.remove(newBoard, row, col)
                col += 1
            row += 1
            
        row = 0
        for r in newBoard:
            col = 0
            for element in r:
                if element == WHITE: 
                    if self.isDead(newBoard, row, col, WHITE):
                        newBoard = self.remove(newBoard, row, col)
                col += 1
            row += 1
        return newBoard
    
    def findDistance(self, start, end):
        total = 0
        total += abs(int(start[0]) - int(end[0]))
        total += abs(int(start[1]) - int(end[1]))
        return total
        
    def findTotalDistance(self, state):
        total = 0
        row = 0
        for r in state:
            col = 0
            for element in r:
                if element == BLACK:
                    row2 = 0
                    for r2 in state:
                        col2 = 0
                        for element2 in r2:
                            if element2 == WHITE: total += self.findDistance(str(row) + str(col), str(row2) + str(col2))
                            col2 += 1
                        row2 += 1
                col += 1
            row +=1
        return total
        
        
    def aStarSearch(self, max_nodes, maxDepth):
        # Performs a-star search
        # Prints the list of solution moves and the solution length

        # Need a dictionary for the frontier and for the expanded nodes
        frontier_nodes = {}
        expanded_nodes = {}
        
        self.starting_state = copy.deepcopy(self.initialState)
        current_state = copy.deepcopy(self.initialState)
        # Node index is used for indexing the dictionaries and to keep track of the number of nodes expanded
        node_index = 0

        # Set the first element in both dictionaries to the starting state
        # This is the only node that will be in both dictionaries
        expanded_nodes[node_index] = {"state": current_state, "parent": "root", "action": "start",
                                   "total_cost": self.findTotalCost(current_state, 0), "depth": 0}
        
        frontier_nodes[node_index] = {"state": current_state, "parent": "root", "action": "start",
                                   "total_cost": self.findTotalCost(current_state, 0), "depth": 0}
        

        failure = False

        # all_nodes keeps track of all nodes on the frontier and is the priority queue. Each element in the list is a tuple consisting of node index and total cost of the node. This will be sorted by the total cost and serve as the priority queue.
        all_frontier_nodes = [(0, frontier_nodes[0]["total_cost"])]

        # Stop when maximum nodes have been considered
        while not failure:

            # Get current depth of state for use in total cost calculation
            current_depth = 0
            for node_num, node in expanded_nodes.items():
                if node["state"] == current_state:
                    current_depth = node["depth"]

            # Find available actions corresponding to current state
            available_actions = self.getAvailableMoves(current_state)


            # Iterate through possible actions 
            for key, value in available_actions.items():
                for action in value:
                    repeat = False
                 
                    
                    # If max nodes reached, break out of loop
                    if node_index >= max_nodes:
                        failure = True
                        print("No Solution Found in first {} nodes generated".format(max_nodes))
                        self.num_nodes_generated = max_nodes
                        return
                    
                    if current_depth >= maxDepth:
                        failure = True
                        print("No Solution Found in first {} layers".format(maxDepth))
                        return
                    
                    
                    # Find the new state corresponding to the action and calculate total cost
                    if self.isValidMove(current_state, key, action):
                        new_state = self.makeMove(current_state, key, action)
                    else:
                        continue
                    new_state_parent = copy.deepcopy(current_state)
                    
                    # Check to see if new state has already been expanded
                    for expanded_node in expanded_nodes.values():
                        if expanded_node["state"] == new_state:
                            if expanded_node["parent"] == new_state_parent:
                                repeat = True

                    # Check to see if new state and parent is on the frontier
                    # The same state can be added twice to the frontier if the parent state is different
                    for frontier_node in frontier_nodes.values():
                        if frontier_node["state"] == new_state:
                            if frontier_node["parent"] == new_state_parent:
                                repeat = True

                    # If new state has already been expanded or is on the frontier, continue with next action     
                    if repeat:
                        continue

                    else:
                        # Each action represents another node generated
                        node_index += 1
                        depth = current_depth + 1

                        # Total cost is path length (number of steps from starting state) + heuristic
                        new_state_cost = self.findTotalCost(new_state, depth)
                        

                        # Add the node index and total cost to the all_nodes list
                        all_frontier_nodes.append((node_index, new_state_cost))

                        # Add the node to the frontier 
                        frontier_nodes[node_index] = {"state": new_state, "parent": new_state_parent,  "total_cost": new_state_cost, "depth": current_depth + 1, "action" : '({}, {}) -> ({}, {})'.format(key[0], key[1], action[0], action[1])}

                # Sort all the nodes on the frontier by total cost
            all_frontier_nodes = sorted(all_frontier_nodes, key=lambda x: x[1])

                # If the number of nodes generated does not exceed max nodes, find the best node and set the current state to that state
            if not failure:
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
                    # Create attributes for the expanded nodes and the frontier nodes
                    self.expanded_nodes = expanded_nodes
                    self.frontier_nodes = frontier_nodes
                    self.num_nodes_generated = node_index + 1
                    
                    
                    for node_num, node in expanded_nodes.items():
                        if self.isComplete(node["state"]):
                            final_node = expanded_nodes[node_num]
# =============================================================================
#                     print(final_node)
#                     for a in final_node['state']:
#                         print(a)
#                     for a in final_node['parent']:
#                         print(a)
# =============================================================================
                    finalresult = self.generate_solution_path(final_node, expanded_nodes, [])
                    # Display the solution path
                    for a in reversed(finalresult):
                        print(a)
                    break 
    
    
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def main():
    gameState = []

    for i in range(SIZE):
        gameState.append(input().split())

    task = input().split()[0]



    if task == 'Massacre': 
        board = game(gameState)
        board.aStarSearch(1000000000000000000000000,1000000000000000)

    else: print('Invalid mode')



main()
        
        