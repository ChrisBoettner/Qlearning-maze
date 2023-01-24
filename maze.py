#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:40:36 2023

@author: chris
"""

import numpy as np

class MazeGame(object):
    def __init__(self) -> None:
        
        self.board = np.array([[1,1,1,1,1,1,1,1],
                               [1,0,0,0,0,1,0,1],
                               [1,1,1,1,0,1,0,1],
                               [1,0,0,1,0,1,0,1],
                               [1,0,0,0,0,1,0,1],
                               [1,0,0,0,0,0,0,1],
                               [1,1,1,1,1,1,1,1]], dtype=np.int8)
        
        self.start_pos = (1,1)
        self.curr_pos  = list(self.start_pos)
        self.goal      = (1,-2) 
        
        self.board[self.goal] = 2
        
    def start(self) -> None:
        self.curr_pos   = list(self.start_pos)
        
        victory_flag = False
        while not victory_flag:
            player_move = input('Which direction do you want to go? ')
            
            if player_move in ['q' , 'quit' , 'exit']:
                print('Ending game.')
                break
            else:
                try:
                    _, victory_flag = self.turn(player_move)
                except ValueError:
                    print('Direction must be up, down, left or right. '
                          'Type \'quit\' to quit game.')
    
    def turn(self, direction:str) -> tuple[bool,bool]:
        self.move(direction)
        death_flag   = self.check_death()
        victory_flag = self.check_win()
        return(death_flag, victory_flag)
                               
    def check_death(self) -> bool:
        death_flag = False
        if self.board[tuple(self.curr_pos)] == 1:
            print('You ran into a wall. Resetting.')
            self.curr_pos = list(self.start_pos)
            death_flag    = True
        return(death_flag)
        
    def check_win(self) -> int:
        victory_flag = False
        if self.board[tuple(self.curr_pos)] == 2:
            print('You won!')
            victory_flag = True
        return(victory_flag)
  
    def move(self, direction:str) -> None:
        match direction:
            case 'u' | 'up':
                self.curr_pos[0] -= 1
            case 'd' | 'down':
                self.curr_pos[0] += 1
            case 'r' | 'right':
                self.curr_pos[1] += 1
            case 'l' | 'left':
                self.curr_pos[1] -= 1
            case _:
                raise ValueError()
      
class Qplayer(object):
    def __init__(self, maze:MazeGame, alpha:float=1,
                 gamma:float=0.5) -> None:  
        
        self.maze    = maze
        self.q_table = np.zeros([np.prod(self.maze.board.shape),4])
        
        self.actions = {1:'u', 2:'d', 3:'l', 4:'r'}
    
    # def train():
        
    def explore(self, survival_reward:int=1, death_reward:int=-5,
                victory_reward:int=100):
        
        state  = self.position_state_mapping(self.maze.curr_pos)
        action = np.random.randint(1,5)
        
        death_flag, victory_flag = self.maze.turn(self.actions[action])
        
        if victory_flag:
            reward = victory_reward
        elif death_flag:
            reward = death_reward
        else:
            reward = survival_reward
        
        self.update_Q(state, action, reward)
        
    # def exploit():
        
    # def play():
        
    # def update_Q():
        
    def position_state_mapping(self, position:list) -> int:
        state = position[0] + position[1] * self.board.shape[0]
        return(state)
    
    
if __name__=='__main__':
    m = MazeGame()
    qplayer = Qplayer(m)