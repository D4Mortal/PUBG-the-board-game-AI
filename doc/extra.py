###############################################################################
def testMemUsage():
    gameState = np.full((SIZE, SIZE), UNOCC, dtype=int)

#    print("show", sys.getsizeof(board(gameState, ((0,0),(0,1)))))
#    print(gameState )
#    print("show", sys.getsizeof(gameState[0]))

    l = [board(gameState, 'bar', WHITE) for i in range(50000000)]
    print(sys.getsizeof(l))

###############################################################################

def testrun(me = 'white'):
     game = Player(me)

     # update board tests
#     move = ((2,5), (2,6))
#     move2 = ((2,4), (3,4))
#     move3 = ((3,5), (1,1))
#     move4 = ((6,6),(5,6))
#     place = (6,5)
#     null_move = None

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
#     print(sys.getsizeof(game.hashTable))
#     print(game.visited)
#     print(game.miniMaxPlace(4))

# # #
#    print(sys.getsizeof(game.hashTable))

    # print("this is the current board state")
    # print(game.node.state)

#    print('place test')
#
#    for i in list(range(0,24,2)):
#        print('game move', i)
#        if i == 12:
#            game.put_piece(2, 4, BLACK)
#            game.put_piece(2,5, WHITE)
#        print('total_turns', game.totalTurns)
#        print(game.state)
#        game.action(i)
#    print("The ideal move would be: {} for turn 127".format(game.node.move))


#    game.firstShrink()
#    print(game.node.state)
#    print(game.node.eval_node())
#
#    game.secondShrink()
#    print(game.node.state)
#    print(game.node.eval_node())

#    game.update(((2, 5), (4, 5)))
#    print(game.node.state)

#    print(game.node.calculateScore())


    # print(game.node.eval_node())


#    r = zorHash(game.node.state, ZOR)
#    print(r)
#    print(hashMove(r, game.node.state, ((3,5), (1,1))))
#
#
#    game.update(move3)              # move3 is ((3,5), (1,1))
#    a = zorHash(game.node.state, ZOR)
#    print(a)
#
#    game.node.state[4,6] = UNOCC      # remove white piece and recalculate hash from scratch
#    a = zorHash(game.node.state, ZOR)
#    print(a)
#
#    r = r^int(ZOR[3, 5, BLACK])     # hash the moves ((3,5), (1,1)) from initial state
#    r = r^int(ZOR[1, 1, BLACK])
#
#    game.node.state[4,6] = WHITE    # put the white piece back in
#    print(hashRemove(r, game.node.state, (4,6))) # compute hash value from removing white piece




#
#testrun()
#testMemUsage()
