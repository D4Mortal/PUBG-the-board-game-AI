# Chirag Rao Sahib      : 836011
# Daniel Hao            : 834496
# Date                  : 
# Python version        : 3.6.4

###############################################################################
import copy
from collections import defaultdict


# simple board class that stores the current board config and the move that brought
# it there

class board():
    def __init__(self, state, move):
        self.state = state
        self.move = move
    
    
    # function that returns the new board object created from the specified move
    def newMakeMove(self, move):
        newState = copy.deepcopy(self.state)
        
        # make changes in the newState according to the moves specified 
        
        newBoard = board(newState, move)
        return newBoard
    
    
    # function that make moves on the current object, changes the current state,
    # does not create a new board
    def _makeMove(self, move):
        
        
        # make changes in the self.state according to the moves specified 
        
        
        return 