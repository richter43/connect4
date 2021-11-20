#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:18:04 2021

@author: foxtrot
"""

import utils.squilibs as c4
import utils.algos as algos
import numpy as np
import random


class Node():

    def __init__(self, state, player, parent=None):
        self.parent = parent
        self.player = player
        self.state = state
        self.score = 0
        self.visits = 0
        self.branches = []
        self.leaf = True
        self.terminal = False

    def add_branch(self, state):
        node = Node(state, -self.player, parent=self)
        self.branches.append(node)
        self.leaf = False
        return node


def select_best(node, C=1):

    if node.leaf:
        return node

    visit_array = np.array(
        [b.visits == 0 and not b.terminal for b in node.branches])

    if any(visit_array):
        idx = np.argmax(visit_array)
        return node.branches[idx]

    score_array = np.array([ucb(b.score, node.visits, b.visits, C) if not b.terminal else -1
                            for b in node.branches])

    idx = np.argmax(score_array)

    return select_best(node.branches[idx])


def expand(node):

    board = node.state
    node.leaf = False

    val_moves = c4.valid_moves(board)

    if len(val_moves) == 0:
        return None

    tmpList = []

    for move in val_moves:
        board_copy = board.copy()
        c4.play(board_copy, move, -node.player)
        tmpNode = Node(board_copy, -node.player, node)

        if c4.four_in_a_row(board_copy, -node.player):
            tmpNode.terminal = True
        tmpList.append(tmpNode)

    if c4.four_in_a_row(board, node.player):
        return node

    node.branches = tmpList

    choice = np.random.choice(tmpList, 1)[0]

    return choice


def simulate(node):

    node.score = algos.sim(node.state.copy(), node.player)
    node.visits += 1

    return


def backprop(node):

    trav_node = node
    while True:
        trav_node = trav_node.parent
        scores = [i.score for i in trav_node.branches]
        trav_node.score = np.sum(scores)
        trav_node.visits += 1
        if trav_node.parent is None:
            break


def ucb(score, par_n, n, C):
    return score/n + C*np.sqrt(2*np.log(par_n)/n)


def place_randomly(board, player):
    play = random.sample(c4.valid_moves(board), 1)[0]
    c4.play(board, play, player)
