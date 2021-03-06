# Artificial Intelligence Project Part B - Playing a Game of *Watch Your Back* (#*pubg the board game*)

### COMP30024 Artificial Intelligence - Semester 1 2018

#### Chirag Rao Sahib      : 836011
#### Daniel Hao            : 834496
#### Date                  : 10/05/2018


##                                Program structure
#### Essential files:
###### board.py        
- contains board() class
- functions in relation to minimax node generation
- includes evaluation function
###### player.py       
- contains Player() class
- includes minimax functions for both phases
###### zobrist_hash.py 
- contains functions related to Zobrist hashing
- explained in creativity section
###### constants.py    
- constants including those learned through external means

#### Non-essential files:
###### minimax_sampler.py
- Generates random board configurations to compute ideal depth per move for given time/space constraints
###### extra.py         
- independent testing of game functions
###### referee.py  
- Created by Matt Farrugia and Shreyash Patodia
- Referee program that simulates game play between 2 gaming playing agents
- How to run: python referee.py *player_agent_1* *player_agent_2* (must be in the same directory as essential files)

###### Note: minimax algorithm has been adapted from AIMA library, using Python 3.6




##                                Search strategy:

After researching Google's AlphaGo AI we explored a new search tree algorithm
called Monte Carlo Tree Search. However as the name suggests, Monte Carlo tree
search makes use of simulated game playouts. Considering the original time and
space constraints this was clearly infeasible. Nevertheless we continued our
research and were inspired by a Chess AI called Stockfish. Stockfish is one of
the best Chess AI's available and at it's core uses minimax with alpha-beta
pruning. Since the average branching factor of Chess is similar to that of
Watch your back (~35 vs ~40). Hence the core of our AI is also minimax with
alpha-beta search. Initially we hard-coded exactly which positions to play
in the placing phase. However upon testing this against other players, we found
our logic was flawed. We decided that minimax for the placing phase is the
smartest choice since it produces dynamic moves in response to the other
player.



 ##                                Evaluation function:

Our evaluation function consists of 5 features.
f1 - simply count of how many player pieces currently on the board
f2 - same as above but for the enemy, ALWAYS negatively weighted
f3 - count internal position score per piece-square tables
f4 - moving phase only, how many legal+safe moves available to play
f5 - counts number of friendly surrounding pieces present

f1 and f2 are simple features that provide a general outlook of the current
state. It is important to separate these features since we want to assign a
greater weight to states that contain more friendly pieces. f3 is inspired
by Chess AI that incorporate piece-square tables. The piece-square tables are
defined in the constants.py file. Each position on the board is scored,
higher values indicate the position is more valuable. The eval function counts
this total score of valuable positions in a given state. f4 is also inspired
by Chess AI that use 'safe mobility'. This is the total number of legal and
safe moves available to the player. States with a higher f4 mean that the AI
can move around more freely whether it be to 'kill' or obtain valuable
positions. Eliot Slater conducted a study (of chess) in which he describes a
definite correlation between player mobility and win rate. Another important
feature that becomes clear after playing Watch your back a few times is the
concept of 'connectedness'. f5 aims to quantify this, by counting the number
of surrounding friendly pieces. This is a defensive tactic where we assume a
lone piece is much more likely to be eliminated by the enemy.

The bounds for the evaluation function features are as follows:

f1 : (0,12)
f2 : (0,12)
f3 : (0,36)
f4 : (0,48)
f5 : (0,48)


##                              Optimizations:

We have made use of a variety of optimisations and techniques commonly found in
game playing agents. Firstly the simplest optimisation is using a numpy array
to represent the board. numpy has a C backend, and hence any operations
involving the board representation is likely to be faster. 
Another simple optimisation is the use of __slots__ in the board() class
for faster attribute access. Loops were reused where possible to minimise
any impact on performance, though readability suffers slightly.

We have also made use of transposition tables in combination with Zobrist
hashing. During minimax node generation, there are likely to be many states
that will be repeated. Zobrist hashing allows us to hash these states and
efficiently store the states themselves and their computed minimax value. This
saves much computation time when the state is encountered again. Zobrist
hashing is extremely efficient as it user XOR operations to update the board,
instead of recomputing the hash each time; essentially only updating any
differences in the board state. We eliminate the possibility of hash collisions
by setting an extremely large integer for each board position.

We did try implementing iterative deepening search in combination with our
minimax search however the main overhead was from generating nodes rather than
evaluation (since this is stored by our transposition table). As an alternative
we decided to use a sampling/simulation program (minimax_sampler.py) to
estimate the ideal search depth in relation to the current branching factor.
(the number of child nodes for both players). Using the total branching factor
overestimates the time usage for different depths, just to be safe and remain
within the given limits. Our results were then stored in the constants.py
(IDEAL_DEPTH) file and is made use of by our game to essentially produce a
minimax search with a dynamic depth.

Note: a constant minimax search depth is used in the placing phase, as a depth
of 3 offers the best balance between time and space usage

As mentioned in the lectures, "Good move ordering improves effectiveness of
pruning" of search tree algorithms. This idea is also similar to the principal
variation search (NegaScout) algorithm. Once the child nodes are generated
we use a cheap (important) evaluation function to sort the nodes. The cheap
evaluation is simply the number of player pieces minus the number of enemy
pieces in the current state. The 'principal variation' is usually the first
state in the sorted list, which has the highest number of friendly pieces. This
allows more efficient pruning since the best moves are unlikely to take place
if we assume the enemy agent is also optimal. This also incorporates some of
the idea of 'quiescence search' in which states with more friendly pieces
(and consequently less enemy pieces) are expanded first. This is a result of
the move ordering (principal variation) mentioned above and helps reduce the
horizon effect.

We also incorporate strict move ordering into our placing phase by aggressively
pruning nodes outside of our defined ORDERMAP. This ORDERMAP is based on the
idea of centre control discussed above.

Again considering our evaluation function it is worth justifying our
piece-square tables defined in constants.py. Firstly it can be argued that the
main strategy for the game is maintaining control of the board centre. This
allows a player to take maximum advantage of the shrinking stages, in which
the enemy will be eliminated if all the center positions are taken.
Accordingly we set the center most pieces of the piece square table to be most
valuable, and decrease in value as we expand outwards. The difference between
PLACEMAP_BLACK, PLACEMAP_WHITE is that positions closer to the 'colours side'
are valued slightly more. Furthermore we assign 0 to positions such as (1,2),
and shrinking areas, where pieces will be eliminated instantly on board shrink.



