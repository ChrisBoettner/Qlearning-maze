#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 07:32:19 2023

@author: chris
"""

from maze import MazeGame
from qagent import QAgent

if __name__ == "__main__":
    maze = MazeGame()
    qagent = QAgent(maze)
