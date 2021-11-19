#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:18:04 2021

@author: foxtrot
"""

import utils.squilibs as c4
from utils.squilibs import montecarlo
import numpy as np
import random
import sys


class Node():

    def __init__(self, state, player, parent=None):
        self.parent = parent
        self.player = player
        self.state = state
        self.score = 0
        self.visits = 0
        self.branches = []
        self.leaf = True

    def add_branch(self, state):
        node = Node(state, -self.player, parent=self)
        self.branches.append(node)
        self.leaf = False
        return node


def select_best(node):

    if node.leaf:
        return node

    score_array = [ucb(b.score, node.visits, b.visits)
                   for b in node.branches]

    idx = np.argmax(score_array)

    return select_best(node.branches[idx])


def expand(node):

    board = node.state
    node.leaf = False

    val_moves = c4.valid_moves(board)

    if c4.four_in_a_row(board, node.player):
        return node

    if len(val_moves) == 0:
        return None

    move = np.random.choice(val_moves, 1)[0]
    board_copy = board.copy()
    c4.play(board_copy, move, -node.player)

    return node.add_branch(board_copy)


def simulate(node):

    node.score = montecarlo(node.state.copy(), node.player)
    node.visits += 1

    return


def backprop(node):

    trav_node = node

    while True:
        trav_node = trav_node.parent
        scores = [i.score for i in trav_node.branches]
        trav_node.score += np.sum(scores)
        trav_node.visits += 1
        if trav_node.parent is None:
            break


def ucb(score, par_n, n, C=1):
    return score/n + C*np.sqrt(2*np.log(par_n)/n)


def place_randomly(board, player):
    play = random.sample(c4.valid_moves(board), 1)[0]
    c4.play(board, play, player)
