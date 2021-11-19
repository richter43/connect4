#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 09:27:29 2021

@author: foxtrot
"""

import utils.squilibs as c4
import numpy as np

BRANCHING_FACTOR = 7
PLAYER_1 = 1
PLAYER_2 = -1
MAX_DEPTH = 5


def minmax(board, player, level):
    '''

    Parameters
    ----------
    board : Numpy array
        DESCRIPTION.
    player : Integer
        Values are either 1 or -1, the player is the one currently playing.
    level : Integer
        Represents the depth at which the recursion is currently at.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    board_moves = c4.valid_moves(board)

    if c4.four_in_a_row(board, 1) or c4.four_in_a_row(board, -1):
        return (None, player*(level+1))
    elif level == 0 or not board_moves:
        return (None, 0)

    ans = (None, -2**32) if player == PLAYER_1 else (None, 2**32)

    for move in board_moves:
        c4.play(board, move, player)
        tmpAns = minmax(board, -player, level - 1)
        c4.take_back(board, move)
        ans = max(ans, (move, -tmpAns[1]), key=lambda x: x[1])

    return ans


def minimax_alphabeta(board, player, level, alpha, beta):
    '''

    Modified version of the minmax which introduces alpha beta pruning

    Parameters
    ----------
    board : Numpy array
        DESCRIPTION.
    player : Integer
        Values are either 1 or -1, the player is the one currently playing.
    level : Integer
        Represents the depth at which the recursion is currently at.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    if c4.four_in_a_row(board, 1) or c4.four_in_a_row(board, -1):
        return (None, player*(level+1))
    elif level == 0 or not c4.valid_moves(board):
        return (None, 0)

    board_moves = c4.valid_moves(board)
    for move in board_moves:
        c4.play(board, move, player)
        tmpAlpha = minimax_alphabeta(
            board, -player, level - 1, (beta[0], -beta[1]), (alpha[0], -alpha[1]))
        c4.take_back(board, move)
        alpha = max(alpha, (move, -tmpAlpha[1]), key=lambda x: x[1])
        if alpha[1] >= beta[1]:
            return beta

    return alpha


def minimax_montecarloeval(board, player, level, alpha, beta):
    '''

    Modified version of the minmax which introduces alpha beta pruning

    Parameters
    ----------
    board : Numpy array
        DESCRIPTION.
    player : Integer
        Values are either 1 or -1, the player is the one currently playing.
    level : Integer
        Represents the depth at which the recursion is currently at.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    if c4.four_in_a_row(board, 1) or c4.four_in_a_row(board, -1):
        return (None, player*(level+1))
    elif not c4.valid_moves(board):
        return (None, 0)
    elif level == 0:
        return (None, player*(c4.montecarlo(board, player)))

    board_moves = c4.valid_moves(board)
    for move in board_moves:
        c4.play(board, move, player)
        tmpAlpha = minimax_montecarloeval(
            board, -player, level - 1, (beta[0], -beta[1]), (alpha[0], -alpha[1]))
        c4.take_back(board, move)
        alpha = max(alpha, (move, -tmpAlpha[1]), key=lambda x: x[1])
        if alpha[1] >= beta[1]:
            return beta

    return alpha
