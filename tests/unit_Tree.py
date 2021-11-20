#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 20:39:11 2021

@author: foxtrot
"""

import unittest
import utils.tree as tree
import numpy as np
import utils.squilibs as c4
import utils.algos as algos

PLAYER = -1


class testMCST(unittest.TestCase):

    def setUp(self):
        self.je = johnLevineExample()
        self.je.branches[0].score = 20
        self.je.branches[0].visits = 1
        tree.backprop(self.je.branches[0])
        self.je.branches[1].score = 10
        self.je.branches[1].visits = 1
        tree.backprop(self.je.branches[1])
        self.assertEqual(self.je.score, 30)
        self.assertEqual(self.je.visits, 2)

    def test_rootisleaf(self):
        root = createNode(0, 0, None, True)
        selected_node = tree.select_best(root)
        self.assertEqual(root, selected_node)

    def test_leavesempty(self):
        root = johnLevineExample()
        root.branches[0].score = 20
        root.branches[0].visits = 1
        tree.backprop(root.branches[0])
        self.assertEqual(root.visits, 1)
        self.assertEqual(root.score, 20)

    def test_selectiononeempty(self):
        root = johnLevineExample()
        root.branches[0].score = 20
        root.branches[0].visits = 1
        selected_node = tree.select_best(root)
        self.assertEqual(selected_node, root.branches[1])

    def test_select_ucb(self):

        selected_node = tree.select_best(self.je, 2)
        self.assertEqual(selected_node, self.je.branches[0])


class testMCST_c4(unittest.TestCase):

    def setUp(self):
        self.board = np.zeros(
            (c4.NUM_COLUMNS, c4.COLUMN_HEIGHT), dtype=np.byte)

    def test_expandsimulate(self):
        root = tree.Node(self.board, -1)
        sel_node = tree.select_best(root)
        self.assertEqual(sel_node, root)
        select_sim = tree.expand(sel_node)
        self.assertEqual(len(root.branches), 7)
        tree.simulate(select_sim)
        tree.backprop(select_sim)
        self.assertEqual(select_sim.visits, 1)
        tree.select_best(root)

    def test_squil(self):
        board = self.board.copy()
        c4.play(board, 3, 1)
        c4.play(board, 0, -1)
        c4.play(board, 4, 1)
        c4.play(board, 0, -1)
        c4.play(board, 5, 1)
        c4.play(board, 0, -1)

        root = tree.Node(board, -1)
        idx = algos.mcst(root, algos.PLAYER_2, 15)

        self.assertEqual(idx, 2)


def johnLevineExample():
    root = createNode(0, 0, None)
    first_leaf = createNode(0, 0, root, True)
    second_leaf = createNode(0, 0, root, True)

    root.branches = [first_leaf, second_leaf]

    return root


def createNode(score, visits, parent, leaf=False):
    node = tree.Node(None, 0)
    node.score = score
    node.visits = visits
    node.parent = parent
    node.leaf = leaf
    return node


if __name__ == '__main__':
    unittest.main()
