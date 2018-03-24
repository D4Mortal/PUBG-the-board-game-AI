<<<<<<< HEAD
# test1
# test daniel
=======
#test chirag
>>>>>>> d5f3c52a8910788b7d17d9b7a48bdf2dd3da35ee
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
CORNER = 'X'

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



def NumOfSurroundingBlack(board, row, col):
    count = 0
    checkCond = {'D':[row+1 < SIZE],
               'U':[row-1 >= 0],
               'R':[col+1 < SIZE],
               'L':[col-1 >= 0]}

    for m in checkCond:
        if checkCond[m][0]:
            posCheck = move(board,row,col, m)
            if posCheck == BLACK: count += 1

    return count


def checkSurr(board, row, col):
    availMoves = 0
    checkCond = {'D':[row+1 < SIZE, row+2 < SIZE],
               'U':[row-1 >= 0, row-2 >= 0],
               'R':[col+1 < SIZE, col+2 < SIZE],
               'L':[col-1 >= 0, col-2 >= 0]}

    for m in checkCond:
        if checkCond[m][0]:
            posCheck = move(board,row,col, m)
            if posCheck == UNOCC: availMoves += 1
            if posCheck == WHITE or posCheck == BLACK:
                if checkCond[m][1]:
                    posCheck2 = move(board,row,col,'2' + m)
                    if posCheck2 == UNOCC: availMoves += 1
    return availMoves



def movePiece(board, colour, start, end):
    board[int(start[0])][int(start[1])] = UNOCC
    board[int(end[0])][int(end[1])] = colour
    return board

def remove(board, position):
    board[int(position[0])][int(position[1])] = UNOCC
    return board

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


# a deadend is a slot where if a white piece moves into it, it will be instantly eliminated
def isDeadEnd(board, rowNum, colNum, rowEnd, colEnd):
    if rowNum == rowEnd and colNum == colEnd:
        return False
    if rowNum + 1 < 8 and rowNum - 1 >= 0:
        if down(board, rowNum, colNum) == BLACK or down(board, rowNum, colNum) == CORNER:
            if up(board, rowNum, colNum) == BLACK or up(board, rowNum, colNum) == CORNER:
                return True;

    if colNum + 1 < 8 and colNum - 1 >= 0:
        if right(board, rowNum, colNum) == BLACK or right(board, rowNum, colNum) == CORNER:
            if left(board, rowNum, colNum) == BLACK or left(board, rowNum, colNum) == CORNER:
                return True;

    return False


def createTree(board, rowStart, colStart, rowEnd, colEnd):
    black = '@'
    white = 'O'
    unoccupied = '-'
    graph = defaultdict(list)
    row = 0


    # builds an adjacency matrix of unoccupied spaces, it excludes unoccupied deadends unless its the goal area, as it would
    # result in the white piece getting eliminated
    # it treats the starting position as unoccupied space it needs to know where it can move to from the starting position
    for line in board:
        col = 0
        for symbol in line:
            if (symbol == UNOCC or (row == rowStart and col == colStart)) and not isDeadEnd(board, row, col, rowEnd, colEnd):

                if row + 1 < 8:
                    if down(board, row, col) == unoccupied and not isDeadEnd(board, row + 1, col, rowEnd, colEnd):
                        graph[str(row) + str(col)].append(str(row + 1) + str(col))

                    elif down(board, row, col) == white or down(board, row, col) == black:
                        if row + 2 < 8 and not isDeadEnd(board, row + 2, col, rowEnd, colEnd):
                            if two_down(board, row, col) == unoccupied:
                                graph[str(row) + str(col)].append(str(row + 2) + str(col))


                if row - 1 >= 0:
                    if up(board, row, col) == unoccupied and not isDeadEnd(board, row - 1, col, rowEnd, colEnd):
                        graph[str(row) + str(col)].append(str(row - 1) + str(col))

                    elif up(board, row, col) == white or up(board, row, col) == black:
                        if row - 2 >= 0 and not isDeadEnd(board, row - 2, col, rowEnd, colEnd):
                            if two_up(board, row, col) == unoccupied:
                                graph[str(row) + str(col)].append(str(row - 2) + str(col))


                if col + 1 < 8 and not isDeadEnd(board, row, col + 1, rowEnd, colEnd):
                    if right(board, row, col) == unoccupied:
                        graph[str(row) + str(col)].append(str(row) + str(col + 1))

                    elif right(board, row, col) == white or right(board, row, col) == black:
                        if col + 2 < 8 and not isDeadEnd(board, row, col + 2, rowEnd, colEnd):
                            if two_right(board, row, col) == unoccupied:
                                graph[str(row) + str(col)].append(str(row) + str(col + 2))


                if col - 1 >= 0 and not isDeadEnd(board, row, col - 1, rowEnd, colEnd):
                    if left(board, row, col) == unoccupied:
                        graph[str(row) + str(col)].append(str(row) + str(col - 1))

                    elif left(board, row, col) == white or left(board, row, col) == black:
                        if col - 2 >= 0 and not isDeadEnd(board, row, col - 2, rowEnd, colEnd):
                            if two_left(board, row, col) == unoccupied:
                                graph[str(row) + str(col)].append(str(row) + str(col - 2))


            col += 1
        row += 1

    return graph


# this function assumes that the provided position must have a solution
def choosePosition(board, row, col):
    target = []

    # if the piece lies on the wall, then it can only be eliminated in one
    # direction.
    if row == 7 or row == 0:
        if col == 1:
            target.append(str(row) + '2')
            return target

        if col == 6:
            target.append(str(row) + '5')
            return target

        # if one of the sides is a deadend, add the other one first, because the order of
        # moving the white piece will matter in this case
        if isDeadEnd(board, row, col - 1, row, col):
            target.append(str(row) + str(col + 1))
            target.append(str(row) + str(col - 1))
        else:
            target.append(str(row) + str(col - 1))
            target.append(str(row) + str(col + 1))
        return target

    elif col == 7 or col == 0:

        if row == 1:
            target.append('2' + str(col))
            return target

        if row == 6:
            target.append('5' + str(col))
            return target

        if isDeadEnd(board, row - 1, col, row, col):
            target.append(str(row + 1) + str(col))
            target.append(str(row - 1) + str(col))
        else:
            target.append(str(row - 1) + str(col))
            target.append(str(row + 1) + str(col))
        return target

    else:
        # this checks if the two opposite are both deadends, if they are, then return the other pair as the
        # chosen position, as the function assumes there is a solution
        if isDeadEnd(board, row, col - 1, row, col) and isDeadEnd(board, row, col + 1, row, col):
            if isDeadEnd(board, row - 1, col, row, col):
                target.append(str(row + 1) + str(col))
                target.append(str(row - 1) + str(col))
            else:
                target.append(str(row - 1) + str(col))
                target.append(str(row + 1) + str(col))
            return target

        if isDeadEnd(board, row - 1, col, row, col) and isDeadEnd(board, row + 1, col, row, col):
            if isDeadEnd(board, row, col - 1, row, col):
                target.append(str(row) + str(col + 1))
                target.append(str(row) + str(col - 1))
            else:
                target.append(str(row) + str(col - 1))
                target.append(str(row) + str(col + 1))
            return target

        # check if there's a black piece around the point, if there is, choose the sides
        # that doesn't have a black piece, as it is impossible to eliminate with a black
        # piece taking up the slot
        if move(board, row, col, 'U') == BLACK or move(board, row, col, 'D') == BLACK:
            if isDeadEnd(board, row, col - 1, row, col):
                target.append(str(row) + str(col + 1))
                target.append(str(row) + str(col - 1))
            else:
                target.append(str(row) + str(col - 1))
                target.append(str(row) + str(col + 1))
            return target

        if move(board, row, col, 'R') == BLACK or move(board, row, col, 'L') == BLACK:
            if isDeadEnd(board, row - 1, col, row, col):
                target.append(str(row + 1) + str(col))
                target.append(str(row - 1) + str(col))
            else:
                target.append(str(row - 1) + str(col))
                target.append(str(row + 1) + str(col))
            return target

        # after all these checks, the 4 directions should all be either empty or contains a white piece
        # if there's already a white piece, only return one target position


        # the very last case is there's nothing surrounding the current position, in this case
        # simply choose the top and bottom as the target, as due to the restriction to the placing
        # phase, the y coordinates would be closer to the middle, creating better pathing
        if isDeadEnd(board, row - 1, col, row, col):
            target.append(str(row + 1) + str(col))
            target.append(str(row - 1) + str(col))
        else:
            target.append(str(row - 1) + str(col))
            target.append(str(row + 1) + str(col))
        return target

    return target



# craete a list of black pieces that needs to be eliminated
# order is determined by the number of surrounding black pieces
# in increasing order
def eliminationList(board):
    order = defaultdict(int)
    row = 0
    for line in board:
        col = 0
        for value in line:
            if board[row][col] == BLACK:
                order[str(row) + str(col)] = NumOfSurroundingBlack(board, row, col)
            col += 1
        row += 1
    return sorted(order, key = order.get)

# choose the closest white piece but exclude the white piece that was 
# used for the oppposite direction
def chooseWhite(board, targetRow, targetCol, excludeRow, excludeCol):
    order = defaultdict(int)
    row = 0
    for line in board:
        col = 0
        for value in line:
            if board[row][col] == WHITE and row != excludeRow and col != excludeCol:
                order[str(row) + str(col)] = abs(targetRow - row) + abs(targetCol - col)
            col += 1
        row += 1
    result = sorted(order, key = order.get)
    position = result[0]
    return int(position[0]), int(position[1])

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


def formatResult(result):
    print(result)
    row = 0
    col = 0
    counter = 0
    if len(result) > 1:
        for coordinate in result:

            if counter > 0:
                print('({}, {}) -> ({}, {})'.format(row, col, coordinate[0], coordinate[1]))
            row = coordinate[0]
            col = coordinate[1]
            counter +=1
            
            
def Massacre(board):
    eliminationOrder = eliminationList(board)
    for black in eliminationOrder:
        usedWhite = '00'
        targets = []
        targets = choosePosition(board, int(black[0]), int(black[1]))
        for t in targets:
            rowStart, colStart = chooseWhite(board, int(t[0]), int(t[1]), int(usedWhite[0]), int(usedWhite[1]))
            startAt = str(rowStart) + str(colStart)
            usedWhite = t
            tree = createTree(board, rowStart, colStart, int(t[0]), int(t[1]))
            final = bfs(tree, startAt, t)
            formatResult(final)
            board = movePiece(board, WHITE, startAt, t)
        board = remove(board, black)



def main():
    gameState = []

    for i in range(SIZE):
        gameState.append(input().split())

    task = input().split()[0]

    if task == 'Moves':
        whiteMoves, blackMoves = testMoves(gameState)
        print('{}\n{} moves'.format(whiteMoves, blackMoves))
    elif task == 'Massacre':
        Massacre(gameState)
    else:
        print('invlid mode')



main()

