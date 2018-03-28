from collections import defaultdict

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



def numOfSurrBlack(board, row, col):
    '''
    count black pieces surrounding current position
    '''
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
    '''
    '''
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
    '''
    '''
    board[int(start[0])][int(start[1])] = UNOCC
    board[int(end[0])][int(end[1])] = colour
    return board

def remove(board, position):
    '''
    '''
    board[int(position[0])][int(position[1])] = UNOCC
    return board

def testMoves(gameState):
    '''
    '''
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


def isDeadEnd(board, rowNum, colNum, rowEnd, colEnd):
    '''
    detects deadends: if a white piece moves into a deadend
                        it will be instantly eliminated
    '''

    if rowNum == rowEnd and colNum == colEnd:
        return False

    if rowNum + 1 < SIZE and rowNum - 1 >= 0:
        if move(board, rowNum, colNum, 'D') == BLACK or move(board, rowNum, colNum, 'D') == CORNER:
            if move(board, rowNum, colNum, 'U') == BLACK or move(board, rowNum, colNum, 'U') == CORNER:
                return True;

    if colNum + 1 < SIZE and colNum - 1 >= 0:
        if move(board, rowNum, colNum, 'R') == BLACK or move(board, rowNum, colNum, 'R') == CORNER:
            if move(board, rowNum, colNum, 'L') == BLACK or move(board, rowNum, colNum, 'L') == CORNER:
                return True;

    return False


def createTree(board, rowStart, colStart, rowEnd, colEnd):
    '''
    - build an adjacency matrix of unoccupied spaces
    - excludes unoccupied deadends unless its the goal area,
        since white piece would be eliminated
    - starting position treated as unoccupied space as it needs to know where
        it can move to from the starting position
    '''

    row = 0
    graph = defaultdict(list)

    for line in board:
        col = 0
        for symbol in line:
            if (symbol == UNOCC or (row == rowStart and col == colStart)) and not isDeadEnd(board, row, col, rowEnd, colEnd):

                if row + 1 < SIZE:
                    if move(board, row, col, 'D') == UNOCC and not isDeadEnd(board, row + 1, col, rowEnd, colEnd):
                        graph[str(row) + str(col)].append(str(row + 1) + str(col))

                    elif move(board, row, col, 'D') == WHITE or move(board, row, col, 'D') == BLACK:
                        if row + 2 < SIZE and not isDeadEnd(board, row + 2, col, rowEnd, colEnd):
                            if move(board, row, col, '2D') == UNOCC:
                                graph[str(row) + str(col)].append(str(row + 2) + str(col))


                if row - 1 >= 0:
                    if move(board, row, col, 'U') == UNOCC and not isDeadEnd(board, row - 1, col, rowEnd, colEnd):
                        graph[str(row) + str(col)].append(str(row - 1) + str(col))

                    elif move(board, row, col, 'U') == WHITE or move(board, row, col, 'U') == BLACK:
                        if row - 2 >= 0 and not isDeadEnd(board, row - 2, col, rowEnd, colEnd):
                            if move(board, row, col, '2U') == UNOCC:
                                graph[str(row) + str(col)].append(str(row - 2) + str(col))


                if col + 1 < SIZE and not isDeadEnd(board, row, col + 1, rowEnd, colEnd):
                    if move(board, row, col, 'R') == UNOCC:
                        graph[str(row) + str(col)].append(str(row) + str(col + 1))

                    elif move(board, row, col, 'R') == WHITE or move(board, row, col, 'R') == BLACK:
                        if col + 2 < SIZE and not isDeadEnd(board, row, col + 2, rowEnd, colEnd):
                            if move(board, row, col, '2R') == UNOCC:
                                graph[str(row) + str(col)].append(str(row) + str(col + 2))


                if col - 1 >= 0 and not isDeadEnd(board, row, col - 1, rowEnd, colEnd):
                    if move(board, row, col, 'L') == UNOCC:
                        graph[str(row) + str(col)].append(str(row) + str(col - 1))

                    elif move(board, row, col, 'L') == WHITE or move(board, row, col, 'L') == BLACK:
                        if col - 2 >= 0 and not isDeadEnd(board, row, col - 2, rowEnd, colEnd):
                            if move(board, row, col, '2L') == UNOCC:
                                graph[str(row) + str(col)].append(str(row) + str(col - 2))


            col += 1
        row += 1

    return graph



def choosePosition(board, row, col):
    '''
    assumes the provided position must have a solution
    '''
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




def eliminationList(board):
    '''
    - generate list of black pieces to be eliminated
    - list order determined by the number of surrounding black pieces
        , in increasing order
    '''
    row = 0
    order = defaultdict(int)

    for line in board:
        col = 0
        for value in line:
            if board[row][col] == BLACK:
                order[str(row) + str(col)] = numOfSurrBlack(board, row, col)
            col += 1
        row += 1

    return sorted(order, key = order.get)




def chooseWhite(board, targetRow, targetCol, excludeRow, excludeCol):
    '''
    choose the closest white piece but exclude the white piece that was
    used for the oppposite direction
    '''
    row = 0
    order = defaultdict(int)

    for line in board:
        col = 0

        for value in line:
            if board[row][col] == WHITE and (str(row) + str(col) != str(excludeRow) + str(excludeCol)):
                order[str(row) + str(col)] = abs(targetRow - row) + abs(targetCol - col)
            col += 1
        row += 1

    result = sorted(order, key = order.get)
    position = result[0]
    return int(position[0]), int(position[1])


def bfs(graph, start, end):
    queue = []  # maintain a queue of paths
    queue.append([start])  # push first path into queue

    while queue:
        path = queue.pop(0)  # get first path from queue
        node = path[-1]  # get  last node from  path
        if node == end: return path   # path found
        # enumerate all adjacent nodes, construct a new path
            # and push it into the queue
        for adjacent in graph.get(node, []):
            newPath = list(path)
            newPath.append(adjacent)
            queue.append(newPath)
        
    
def isDead(board, row, col):
    if row == 0 or row == 7:
        if move(board, row, col, "L") == BLACK or  move(board, row, col, "L") == CORNER:
            if move(board, row, col, "R") == BLACK or  move(board, row, col, "R") == CORNER:
                return True
    if col == 0 or row == 7:
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
    return False

def formatResult(result):
    row = 0
    col = 0
    counter = 0

    if len(result) > 1:
        for pos in result:
            if counter > 0:
                print('({}, {}) -> ({}, {})'.format(col, row, pos[1], pos[0]))

            row = pos[0]
            col = pos[1]
            counter += 1


def massacre(board):
    eliminationOrder = eliminationList(board)

    for black in eliminationOrder:
        usedWhite = '00'
        targets = []
        targets = choosePosition(board, int(black[0]), int(black[1]))


        for t in targets:
            rowStart, colStart = chooseWhite(board, int(t[0]), int(t[1]), int(usedWhite[0]), int(usedWhite[1]))
            startAt = str(rowStart) + str(colStart)
            tree = createTree(board, rowStart, colStart, int(t[0]), int(t[1]))
            final = bfs(tree, startAt, t)
            formatResult(final)
            board = movePiece(board, WHITE, startAt, t)
#            printBoard(board)
            usedWhite = t

        board = remove(board, black)
        if isDead(board, int(t[0]), int(t[1])):
            board = remove(board, t)

def printBoard(board):
    for rows in board:
        print(rows)
        
        
def main():
    gameState = []

    for i in range(SIZE):
        gameState.append(input().split())

    task = input().split()[0]

    if task == 'Moves':
        whiteMoves, blackMoves = testMoves(gameState)
        print('{}\n{}'.format(whiteMoves, blackMoves))

    elif task == 'Massacre': massacre(gameState)

    else: print('Invalid mode')



main()

