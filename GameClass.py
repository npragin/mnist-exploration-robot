__author__ = 'Caleytown'
import numpy as np

class Game:
    def __init__(self, truthMap, digit, navigator, robot):
        """
        Inputs:
        map_gt: The map to be explored
        digit: the number 0-9 that the map is actually of
        navigator: a class with a getAction(robot, map) function that generates actions 
                   for the robot to take given the current map
                   actions should have the form "left", "right", "up", or "down"
        robot: the Robot object from RobotClass.py
        """
        self.truthMap = truthMap
        self.navigator = navigator
        self.robot = robot
        self._digit = digit
        self._goal, self._wrong_goals = self._get_goal(digit)

        self.exploredMap = np.ones(truthMap.shape)*128
        self._updateMap(self.robot, self.exploredMap, self.truthMap)
        self.score = 0
        self.iterations = 0


    def tick(self):
        self.iterations += 1
        # Generate an action for the robot 
        action = self.navigator.getAction(self.robot, self.exploredMap)
        # Move the robot
        self.robot.move(action)
        # Update the explored map based on robot position
        self._updateMap(self.robot, self.exploredMap, self.truthMap)
        # Check if we are at the goal and update the score
        if self.robot.getLoc() == self._goal:
            self.score += 100
            return True
        else:
            if self.robot.getLoc() in self._wrong_goals:
                self.score -= 400
            else:
                self.score -= 1
            return False

    def resetGame(self):
        self.iterations = 0
        self.score = 0

    def getScore(self):
        return self.score

    def getIteration(self):
        return self.iterations

    def _updateMap(self, robot, exploredMap, truthMap):

        # Sanity check the robot is in bounds
        if robot.getLoc()[0] > 27 or robot.getLoc()[0] < 0 or robot.getLoc()[1] < 0 or robot.getLoc()[1] > 27:
            raise ValueError(f"Robot has left the map. It is at position: {robot.getLoc()}, outside of the (0-27, 0-27) map boundary")

        # The robot can see all the squares around it
        # So iterate through all nearby squares
        for row_inc in range(-1, 2):
            for col_inc in range(-1, 2):
                col = robot.getLoc()[1] + col_inc
                row = robot.getLoc()[0] + row_inc
                # if the square is off the map, do nothing
                if row > 27 or row < 0 or col < 0 or col > 27:
                    continue
                # Otherwise update the explored map with the actual value of the map
                else:
                    exploredMap[col, row] = truthMap[col, row]

    def _get_goal(self, digit):
        """
            Returns a tuple containing
            - the goal location based on the digit
            - the remaining goal locations 
    
            Note: You should not need to use this function in your code
        """
        goals = [(0, 27), (27, 27), (27, 0)]
        if digit in range(0,3):
            goal = goals.pop(0)
        elif digit in range(3, 6):
            goal = goals.pop(1)
        elif digit in range(6,10):
            goal = goals.pop(2)
        else:
            raise ValueError("Bad digit input: "+str(digit))

        return goal, goals