#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:38:02 2021

@author: foxtrot
"""

import utils.squilibs as c4
from utils.squilibs import montecarlo
import utils.tree as t
import random
import numpy as np

PLAYER_1 = 1
PLAYER_2 = -1

BRANCHING_FACTOR = 2
MAX_DEPTH = 15


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
    for _ in range(100):
        best_node = t.select_best(root)
        t.expand(best_node, player)
        won = t.simulate(best_node, player)
        best_node.rewards += won
        player = -player

    print(best_node.state)

    ptr = best_node.parent

    while ptr.player != 0:
        print(best_node.state)


def _minmax(board, player, level):

    if level == MAX_DEPTH or not c4.valid_moves(board):
        return 0

    if c4.four_in_a_row(board, 1):
        return 1
    elif c4.four_in_a_row(board, -1):
        return -1

    board_moves = np.array(c4.valid_moves(board))

    val = -2**32 if player == PLAYER_1 else 2**32

    for i in range(BRANCHING_FACTOR):

        board_copy = board.copy()
        if len(board_moves) == 0:
            break
        move = np.random.choice(board_moves, 1)[0]
        board_moves = board_moves[board_moves != move]
        c4.play(board_copy, move, player)
        sol = _minmax(board_copy, -player, level + 1)
        val = max(val, sol) if player == PLAYER_1 else min(val, sol)

    return val


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

c4.play(board, 3, 1)
c4.play(board, 0, -1)
c4.play(board, 4, 1)
c4.play(board, 0, -1)
# c4.play(board, 5, 1)

montecarlo_game(board, PLAYER_1)

# print(_minmax(board, PLAYER_1, 0))

# root = t.Node(board, 0)

# mcst(root, PLAYER_1)
