__author__ = 'Caleytown'

import numpy as np
from PIL import Image
from RobotClass import Robot
from GameClass import Game
from RandomNavigator import RandomNavigator
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork
from GreedyNavigator import GreedyNavigator
from EntropyNavigator import EntropyNavigator
import time

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


# Create a Map Class Object
map = Map()

# Create a Robot that starts at (0,0)
# The Robot Class stores the current position of the robot
# and provides ways to move the robot 
robot = Robot(0, 0)

# Initialize the navigator
# navigator = GreedyNavigator(map.map)
navigator = EntropyNavigator(map.map)

# Create a Game object, providing it with the map data, the goal location of the map, the navigator, and the robot
game = Game(map.map, map.number, navigator, robot)


num_trials = 100
s = 0
t = 0
timer = time.time()

for m in range(0, num_trials):
    map.getNewMap()
    navigator.resetNavigator()
    robot.resetRobot()
    game = Game(map.map, map.number, navigator, robot)

    # This loop runs the game for 1000 ticks, stopping if a goal is found.
    for x in range(0, 1000):
        found_goal = game.tick()
        print(f"{game.getIteration()}: Robot at: {robot.getLoc()}, Score = {game.getScore()}")
        if found_goal:
            print(f"Found goal at time step: {game.getIteration()}!")
            break
    print(f"Final Score: {game.score}")
    s += game.score
    t += game.getIteration()

    im = Image.fromarray(np.uint8(game.exploredMap)).show()

timer = time.time() - timer
print(f"Average score: {s/num_trials}")
print(f"Average time steps: {t/num_trials}")
print(f"Average # of seconds per trial: {timer/num_trials}")
