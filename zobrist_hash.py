# Chirag Rao Sahib      : 836011
# Daniel Hao            : 834496
# Date                  : 10/05/2018
# Python version        : 3.6.4

# Module                : zobrist_hash.py

###############################################################################

import copy
import numpy as np
from constants import SIZE, UNOCC

###############################################################################

def init_table():
    '''
    initialise zobrist hashing table
    '''
    zob_table = np.empty((SIZE, SIZE, 5))
    # 5, since 5 possible values for each position

    for i in range(SIZE):
        for j in range(SIZE):
            for k in range(5):
                zob_table[i,j,k] = np.random.randint(0, 1e19, dtype=np.uint64)

    return zob_table

###############################################################################

def zor_hash(table, state):
    '''
    ***
    '''
    value = 0

    for i in range(SIZE):
        for j in range(SIZE):
            if state[i, j] != UNOCC:
                piece = state[i, j]
                value = value^int(table[i, j, piece])

    return value

###############################################################################

def hash_mv(table, hash_val, colour, action):
    '''
    update hash table with action (uses XOR operations)
    '''
    new_hash = copy.copy(hash_val)
    new_hash = new_hash^int(table[action[0][0], action[0][1], colour])
    new_hash = new_hash^int(table[action[1][0], action[1][1], colour])

    return new_hash

###############################################################################

def hash_rm(table, hash_val, colour, position):
    '''
    update hash table with removed colour from position (uses XOR)
    '''
    new_hash = copy.copy(hash_val)
    new_hash = new_hash^int(table[position[0], position[1], colour])

    return new_hash

###############################################################################
