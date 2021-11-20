#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:38:02 2021

@author: foxtrot
"""

import utils.squilibs as c4
import utils.algos as algos
import utils.tree as tree
import numpy as np

board = np.zeros((c4.NUM_COLUMNS, c4.COLUMN_HEIGHT), dtype=np.byte)

MAX_TIME = 5
p = algos.PLAYER_2

for i in range(10):
    root = tree.Node(board, -p)  # Root should contain player who last played
    idx = algos.mcts(root, p, MAX_TIME)
    c4.play(board, idx, p)
    print(board)
    p = -p
