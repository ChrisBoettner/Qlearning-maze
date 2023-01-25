#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:40:36 2023

@author: chris
"""
import numpy as np
from typing import Any


class MazeGame:
    def __init__(self) -> None:
        '''
        A simple traversable maze game.
        '''
        # setup maze
        self.board = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 0, 0, 0, 0, 1, 0, 1],
                               [1, 1, 1, 1, 0, 1, 0, 1],
                               [1, 0, 0, 1, 0, 1, 0, 1],
                               [1, 0, 0, 0, 0, 1, 0, 1],
                               [1, 0, 0, 0, 0, 0, 0, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        self.start_pos = (1, 1)
        self.target = (1, -2)
        self.board[self.target] = 2

        # initilize player location
        self.curr_pos = list(self.start_pos)

    def start(self) -> None:
        '''
        Start the game.
        '''
        # reset location
        self.reset()

        # loop over turns until victory or quit.
        victory_flag = False
        while not victory_flag:
            player_move = input("Which direction do you want to go? ")

            if player_move in ['q', "quit", "exit"]:
                print('Ending game.')
                break
            else:
                try:
                    _, victory_flag = self.turn(player_move)
                except ValueError:
                    print("Direction must be up, down, left or right. "
                          "Type \'quit\' to quit game.")

    def reset(self) -> None:
        '''
        Reset player location to start
        '''
        self.curr_pos = list(self.start_pos)

    def turn(self, direction: str, **kwargs: Any) -> tuple[bool, bool]:
        '''
        A turn of the game.
        '''
        self.move(direction)
        death_flag = self.check_death(**kwargs)
        victory_flag = self.check_win(**kwargs)
        return death_flag, victory_flag

    def check_death(self, silent: bool = False) -> bool:
        '''
        Check if player ran into wall.
        '''
        death_flag = False
        if self.board[tuple(self.curr_pos)] == 1:
            if not silent:
                print("You ran into a wall. Resetting.")
            self.curr_pos = list(self.start_pos)
            death_flag = True
        return death_flag

    def check_win(self, silent: bool = False) -> bool:
        '''
        Check if player won.
        '''
        victory_flag = False
        if self.board[tuple(self.curr_pos)] == 2:
            if not silent:
                print("You won!")
            victory_flag = True
            self.reset()
        return victory_flag

    def move(self, direction: str) -> None:
        '''
        Input - Movement logic.
        '''
        match direction:
            case 'u' | "up":
                self.curr_pos[0] -= 1
            case 'd' | "down":
                self.curr_pos[0] += 1
            case 'r' | "right":
                self.curr_pos[1] += 1
            case 'l' | "left":
                self.curr_pos[1] -= 1
            case _:
                raise ValueError()
