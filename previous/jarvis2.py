from collections import defaultdict
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


def right(board, row, column):
    return board[row][column + 1]

def two_right(board, row, column):
    return board[row][column + 2]

def left(board, row, column):
    return board[row][column - 1]

def two_left(board, row, column):
    return board[row][column - 2]

def down(board, row, column):
    return board[row + 1][column]

def two_down(board, row, column):
    return board[row + 2][column]

def up(board, row, column):
    return board[row - 1][column]

def two_up(board, row, column):
    return board[row - 2][column]

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





def createTree(board, rowNum, colNum):
    black = '@'
    white = 'O'
    unoccupied = '-'
    graph = defaultdict(list)
    row = 0
    
    for line in board:
        col = 0
        for symbol in line:
            if symbol == UNOCC or (row == rowNum and col == colNum):
                
                if row + 1 < 8:
                    if down(board, row, col) == unoccupied:
                        graph[str(row) + str(col)].append(str(row + 1) + str(col))
                
                    elif down(board, row, col) == white or down(board, row, col) == black:
                        if row + 2 < 8:
                            if two_down(board, row, col) == unoccupied:
                                graph[str(row) + str(col)].append(str(row + 2) + str(col))
                
                
                if row - 1 >= 0:
                    if up(board, row, col) == unoccupied:
                        graph[str(row) + str(col)].append(str(row - 1) + str(col))
        
                    elif up(board, row, col) == white or up(board, row, col) == black:
                        if row - 2 >= 0:
                            if two_up(board, row, col) == unoccupied:
                                graph[str(row) + str(col)].append(str(row - 2) + str(col))  
                
                
                if col + 1 < 8:
                    if right(board, row, col) == unoccupied:
                        graph[str(row) + str(col)].append(str(row) + str(col + 1))
        
                    elif right(board, row, col) == white or right(board, row, col) == black:
                        if col + 2 < 8:
                            if two_right(board, row, col) == unoccupied:
                                graph[str(row) + str(col)].append(str(row) + str(col + 2))
                                    
                
                if col - 1 >= 0:
                    if left(board, row, col) == unoccupied:
                        graph[str(row) + str(col)].append(str(row) + str(col - 1))
        
                    elif left(board, row, col) == white or left(board, row, col) == black:
                        if col - 2 >= 0:
                            if two_left(board, row, col) == unoccupied:
                                graph[str(row) + str(col)].append(str(row) + str(col - 2))
                
            
            col += 1
        row += 1
    
    
    
    return graph



def bfs(graph, start, end):
    # maintain a queue of paths
    queue = []
    # push the first path into the queue
    queue.append([start])
    while queue:
        # get the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        # path found
        if node == end:
            return path
        # enumerate all adjacent nodes, construct a new path and push it into the queue
        for adjacent in graph.get(node, []):
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)


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

    else:
        print('invlid mode')
    graph = createTree(gameState, 2, 5)
 
    print(bfs(graph, '25', '57'))
    
main()

