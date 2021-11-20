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

p = algos.PLAYER_2


MAX_TIME = 5
root = tree.Node(board, p)
idx = algos.mcst(root, p, MAX_TIME)

score_array = [b.score/b.visits for b in root.branches]
idx = np.argmax(score_array)
