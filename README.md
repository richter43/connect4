# Connect 4

Implementation of Montecarlo and Minimax algorithms in the game of Connect 4

# Guidelines

The code is divided in three files folders:

## Utils

* squilibs.py: Libraries made by professor Squillero
* algos.py: Contains my implementation of various algorithms, among them:
  * Minimax.
  * Minimax with alpha-beta pruning.
  * Minimax with alpha-beta pruning and using the montecarlo evaluation as heuristic.
  * Montecarlo evaluation maximizer.
  * Montecarlo Tree Search algorithm with a timer as maximum execution
  * Montecarlo Tree Search Modified Selection with minimax algorithm in the selection phase.
* mcts.py: Contains the implementation of Monte Carlo Tree Search's components (Selection, Expansion, Simulation, Backpropagation) plus some variation (Selection using minimax with alpha-beta pruning)
* tree.py: Class implementation of nodes for creation of a tree

## Tests

Unit tests for the various functions defined in the utils section, contains dummy cases which their results are asserted.

## Main
* sol.py: Contains an iteration of the best solution using MCTS-MS 

s281568 - Samuel Oreste Abreu
