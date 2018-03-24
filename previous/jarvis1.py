BLACK = '@'
WHITE = 'O'
UNOCC = '-'
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
    return board[row + MODS[dir][0]][col + MODS[dir][1]]


def checkSurr(board, row, col):
    availMoves = 0
    checkCond = {'D':[row+1 < SIZE, row+2 < SIZE],
               'U':[row-1 >= 0, row-2 >= 0],
               'R':[col+1 < SIZE, col+2 < SIZE],
               'L':[col-1 < SIZE, col-2 < SIZE]}

    for m in checkCond:
        if checkCond[m][0]:
            posCheck = move(board,row,col, m)
            if posCheck == UNOCC: availMoves += 1
            if posCheck == WHITE or posCheck == BLACK:
                if checkCond[m][1]:
                    posCheck2 = move(board,row,col,'2' + m)
                    if posCheck2 == UNOCC: availMoves += 1
    return availMoves

def testMoves(gameState):
    whiteMoves = 0
    blackMoves = 0
    row = 0

    for line in gameState:
        col = 0
        for symbol in line:
            if symbol == WHITE:
                whiteMoves += checkSurr(gameState, row , col)
            if symbol == BLACK:
                blackMoves += checkSurr(gameState, row , col)
            col += 1
        row += 1
    return whiteMoves, blackMoves



def main():
    gameState = []

    for i in range(SIZE):
        gameState.append(input().split())

    task = input().split()[0]

    if task == 'Moves':
        whiteMoves, blackMoves = testMoves(gameState)
        print('{}\n{}'.format(whiteMoves, blackMoves))
    if task == 'Massacre':
        print('fuckoff')


main()
