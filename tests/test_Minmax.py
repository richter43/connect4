#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 08:55:21 2021

@author: foxtrot
"""
import unittest
import utils.squilibs as c4
import numpy as np
import utils.algos as algos


class TestMinMax(unittest.TestCase):

    def setUp(self):
        self.board = np.zeros(
            (c4.NUM_COLUMNS, c4.COLUMN_HEIGHT), dtype=np.byte)

        c4.play(self.board, 3, 1)
        c4.play(self.board, 0, -1)
        c4.play(self.board, 4, 1)
        c4.play(self.board, 0, -1)
        c4.play(self.board, 5, 1)
        c4.play(self.board, 0, -1)

    def test_minimaxalphabeta(self):

        answer = algos.minimax_alphabeta(self.board, algos.PLAYER_1, 5,
                                         (None, -1000), (None, 1000))
        self.assertTrue(answer == (2, 5))

    def test_minimax(self):

        answer = algos.minmax(self.board, algos.PLAYER_1, 5)
        print(answer)
        self.assertTrue(answer == (2, 5))


if __name__ == '__main__':
    unittest.main()
