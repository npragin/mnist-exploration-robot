__author__ = 'Caleytown'

class Robot:
    def __init__(self, xLoc, yLoc, xLim=(0,27), yLim=(0,27)):
        self.xLim = xLim
        self.yLim = yLim
        self.start_loc = xLoc, yLoc
        self.resetRobot()

    def resetRobot(self):
        self.xLoc = self.start_loc[0]
        self.yLoc = self.start_loc[1]

    def checkValidLoc(self, x, y):
        return x >= self.xLim[0] and x <= self.xLim[1] and y >= self.yLim[0] and y <= self.yLim[1]

    def _setLoc(self, x, y):
        if self.checkValidLoc(x,y):
            self.xLoc = x
            self.yLoc = y
        else:
            raise ValueError(f"Failed to set robot to {(x,y)} outside of bounds\nx limits: {self.xLim}\ny limits: {self.yLim}")

    def getLoc(self):
        return (self.xLoc, self.yLoc)

    def checkValidMove(self, direction, updateState=False):
        """ Checks if the direction is valid

        direction (str): "left", "right", "up", "down" directions to move the robot
        updateState (bool): if True, function also moves the robot if direction is valid
                            otherwise, only perform validity check without moving robot
        """
        if direction == 'left':
            valid = self.xLoc-1 >= self.xLim[0]
            if valid and updateState:
                self.xLoc -= 1

        elif direction == 'right':
            valid = self.xLoc+1 <= self.xLim[1]
            if valid and updateState:
                self.xLoc += 1

        elif direction == 'down':
            valid = self.yLoc+1 <= self.yLim[1]
            if valid and updateState:
                self.yLoc += 1

        elif direction == 'up':
            valid = self.yLoc-1 >= self.yLim[0]
            if valid and updateState:
                self.yLoc -= 1
        else:
            raise ValueError(f"Robot received invalid direction: {direction}!")

        return valid


    def move(self, direction):
        """ Move the robot while respecting bounds"""
        self.checkValidMove(direction, updateState=True)
