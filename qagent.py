#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 07:29:10 2023

@author: chris
"""
import yaml
from tqdm import tqdm
import numpy as np
from typing import Any

from maze import MazeGame


class QAgent:
    def __init__(self, maze: MazeGame) -> None:
        '''
        A Q-learning agent built to learn to solve the maze game.

        Configuration:
            LEARNING_RATE:
                Determines to what extent newly acquired information overrides
                old information. The default is 1.
            DISCOUNT_FACTOR:
                Determines the importance of future rewards. The
                default is 0.5.
            EPSILON_START:
                Probability of choosing explore over exploit strategy at
                beginning of training. The default is 0.8 .
            EPSILON_END:
                Probability of choosing explore over exploit strategy at
                end of training. The default is 0.1 .

            SURVIVAL_REWARD:
                Reward for surviving turn. The default is 1.
            DEATH_PENALTY:
                Penalty for dying. The default is -5.
            VICTORY_REWARD:
                Reward for winning the game. The default is 100.
        '''
        # load configuration
        self.name = "QAGENT"
        self.load_configuration()

        # setup maze and Q-table
        self.maze = maze
        self.q_table = np.zeros([np.prod(self.maze.board.shape), 4])

        # map game actions to integers
        self.actions = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

    def train(self, num_episodes=100, call_limit=int(1e+8)) -> None:
        '''
        Train agent by playing num_episodes of the game. Breaks if either
        num_episodes or call_limit is reached.
        '''

        call_counter = 0
        for episode in tqdm(range(num_episodes)):
            # reset game
            victory_flag = False
            self.maze.reset()

            while not victory_flag:
                call_counter += 1
                # select and execute strategy
                strategy = np.random.rand()
                epsilon = self.epsilon_decay(episode, num_episodes)

                if strategy <= epsilon:
                    victory_flag = self.explore(silent=True)
                else:
                    victory_flag = self.exploit(silent=True)
            # break case to avoid infinite loop
            if call_counter == call_limit:
                raise RuntimeError('Call limit reached.')

    def play(self, print_path: bool = False, call_limit=100) -> None:
        '''
        Play game using current policy defined by Q-table.
        '''
        call_counter = 0
        victory_flag = False

        while not victory_flag:
            if print_path:
                print(self.maze.curr_pos)
            victory_flag = self.exploit(silent=False)

            call_counter += 1
            # break case to avoid infinite loop
            if call_counter == call_limit:
                print('\nCall limit reached. Policy does not appear to lead '
                      'to target.')
                self.maze.reset()
                break

    def explore(self, **kwargs: Any) -> bool:
        '''
        Explore the game by performing random moves and receiving the
        corresponding rewards. Returns victory_flag.
        '''
        state = self.position_state_mapping(self.maze.curr_pos)
        action = np.random.randint(4)

        death_flag, victory_flag = self.maze.turn(self.actions[action],
                                                  **kwargs)
        if victory_flag:
            reward = self.config["VICTORY_REWARD"]
        elif death_flag:
            reward = self.config["DEATH_PENALTY"]
        else:
            reward = self.config["SURVIVAL_REWARD"]

        self.update_Q(state, action, reward)

        return victory_flag

    def exploit(self, **kwargs: Any) -> bool:
        '''
        Exploit current policy by taking the action with largest value in
        Q-table. Returns victory_flag.
        '''
        state = self.position_state_mapping(self.maze.curr_pos)
        action = np.argmax(self.q_table[state, :]).astype(int)

        _, victory_flag = self.maze.turn(self.actions[action], **kwargs)
        return victory_flag

    def update_Q(self, state: int, action: int, reward: int) -> None:
        '''
        Update value in Q-table according to Bellmann equation.
        '''
        # current Q value weighted by learning rate
        current_q_term = ((1 - self.config["LEARNING_RATE"])
                          * self.q_table[state, action])
        # reward gained from current state and action weighted by learning rate
        reward_term = self.config["LEARNING_RATE"] * reward

        # maximum reward that can be obtained from new state
        # (weighted by learning rate and discount factor)
        new_state = self.position_state_mapping(self.maze.curr_pos)
        future_q_term = (self.config["LEARNING_RATE"]
                         * self.config["DISCOUNT_FACTOR"]
                         * np.amax(self.q_table[new_state, :]))

        # update Q table
        self.q_table[state, action] = (current_q_term + reward_term
                                       + future_q_term)

    def position_state_mapping(self, position: list[int]) -> int:
        '''
        Maps a position in the maze to the corresponding state in the
        Q-table.
        '''
        state = position[0] + position[1] * self.maze.board.shape[0]
        return state

    def epsilon_decay(self, episode: int, num_episodes: int):
        '''
        Calculate epsilon based on episode. Linear decay.
        '''
        intercept = self.config["EPSILON_START"]
        slope = (self.config["EPSILON_END"] - intercept) / num_episodes
        return slope * episode + intercept

    def load_configuration(self) -> None:
        '''
        Load config from config.yaml file.
        '''
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        self.config = config[self.name]

    def configure(self, configuration: dict[str, float]) -> None:
        '''
        Change configuration of parameter for learning algorithm.
        '''
        for key, value in configuration.items():
            if key in self.config.keys():
                self.config[key] = value
            else:
                raise KeyError(f"{key} not a valid item in the configuration.")


class DQAgent(QAgent):
    def __init__(self, maze: MazeGame) -> None:
        '''
        A Q-learning agent built to learn to solve the maze game.

        Configuration:
            LEARNING_RATE:
                Determines to what extent newly acquired information overrides
                old information. The default is 1.
            DISCOUNT_FACTOR:
                Determines the importance of future rewards. The
                default is 0.5.
            EPSILON_START:
                Probability of choosing explore over exploit strategy at
                beginning of training. The default is 0.8 .
            EPSILON_END:
                Probability of choosing explore over exploit strategy at
                end of training. The default is 0.1 .

            SURVIVAL_REWARD:
                Reward for surviving turn. The default is 1.
            DEATH_PENALTY:
                Penalty for dying. The default is -5.
            VICTORY_REWARD:
                Reward for winning the game. The default is 100.
        '''
        # load configuration
        self.name = "DQAGENT"
        self.load_configuration()

        # setup maze and Q-table
        self.maze = maze

        # map game actions to integers
        self.actions = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    