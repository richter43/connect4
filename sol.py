#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:38:02 2021

@author: foxtrot
"""

import utils.squilibs as c4
import utils.algos as algos
from utils.squilibs import montecarlo
import utils.tree as t
import random
import numpy as np

PLAYER_1 = 1
PLAYER_2 = -1
MAX_DEPTH = 5


def dummy_plays(board, turns):
    for _ in range(turns):
        place_randomly(board, PLAYER_1)
        place_randomly(board, PLAYER_2)
    print(board)


def place_randomly(board, player):
    play = random.sample(c4.valid_moves(board), 1)[0]
    c4.play(board, play, player)


def mcst(root, player):
    best_node = None

    best_node = t.select_best(root)
    sim_node = t.expand(best_node, player)
    t.simulate(sim_node, player)
    t.backprop(sim_node)

    mcst(root, -player)


def montecarlo_game(board, player):

    prob = 0

    while not c4.four_in_a_row(board, player):
        best_move = -1

        for move in c4.valid_moves(board):
            board_copy = board.copy()
            c4.play(board_copy, move, player)
            new_prob = player*montecarlo(board_copy, player)
            if new_prob > prob:
                best_move = move
                prob = new_prob
        c4.play(board, best_move, player)
        print(board)
        moves = c4.valid_moves(board)
        if len(moves) == 0:
            break
        cont_move = np.random.choice(moves, 1)[0]
        c4.play(board, cont_move, -player)


board = np.zeros((c4.NUM_COLUMNS, c4.COLUMN_HEIGHT), dtype=np.byte)

p = algos.PLAYER_1

while len(c4.valid_moves(board)) != 0 and not c4.four_in_a_row(board, 1) and not c4.four_in_a_row(board, -1):
    ans = algos.minimax_montecarloeval(board, p, 1, (0, -1000), (0, 1000))
    c4.play(board, ans[0], p)
    p = -p
    print(board)


# montecarlo_game(board, PLAYER_1)

# print(_minmax(board, PLAYER_1, 0))

# root = t.Node(board, 0)

# mcst(root, PLAYER_1)
