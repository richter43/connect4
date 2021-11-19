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

PLAYER = -1


# class TestTreeMethods(unittest.TestCase):

#     def setUp(self):
#         board = np.zeros(
#             (c4.NUM_COLUMNS, c4.COLUMN_HEIGHT), dtype=np.byte)
#         self.root = tree.Node(board, PLAYER)

#     def test_mcst_root(self):
#         sel = tree.select_best(self.root)
#         self.assertTrue(sel == self.root)
#         sel = tree.expand(self.root)
#         tree.simulate(sel)
#         tree.backprop(sel)
#         self.assertTrue(self.root.visits == 1)
#         sel = tree.select_best(self.root)


class TestMCST(unittest.TestCase):
    def setUp(self):
        self.root = wikiExample()

    def test_mcst_root(self):
        sel = tree.select_best(self.root)
        tree.backprop(sel)
        self.assertTrue(self.root.visits == 1)
        sel = tree.select_best(self.root)


# class TestTreeBackprops(unittest.TestCase):

#     def setUp(self):
#         self.root = tree.Node(None, PLAYER)
#         self.root.add_branch(None)


def wikiExample():

    root = createNode(11, 21, None)
    firstCol = createNode(7, 10, root)
    tmpNode = createNode(1, 6, firstCol)

    tmpNode.branches = [createNode(
        2, 3, tmpNode, True), createNode(3, 3, tmpNode, True)]

    firstCol.branches = [createNode(2, 4, firstCol, True), tmpNode]

    thirdCol = createNode(3, 8, root)
    thirdCol.branches = [createNode(1, 2, thirdCol, True), createNode(
        2, 3, thirdCol, True), createNode(2, 3, thirdCol, True)]

    root.branches = [firstCol, createNode(0, 3, root, True), thirdCol]
    return root


def createNode(score, visits, parent, leaf=False):
    node = tree.Node(None, 0)
    node.score = score
    node.visits = visits
    node.parent = parent
    node.leaf = leaf
    return node


# class TestTreeSelection(unittest.TestCase):
#     def setUp(self):
#         board = np.zeros(
#             (c4.NUM_COLUMNS, c4.COLUMN_HEIGHT), dtype=np.byte)
if __name__ == '__main__':
    unittest.main()
