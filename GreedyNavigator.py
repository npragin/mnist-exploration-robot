import numpy as np
from networkFolder.functionList import WorldEstimatingNetwork, DigitClassificationNetwork

class GreedyNavigator:
    """
    Goes to the brightest unvisited point in the map
    If all points have a brightness of 0, go to the neighbor with most unseen neighbors
    TODO: If we reveal the whole number and the prediction is wrong, we get stuck in a loop.
        Need to encode a way into prediction to not predict numbers that we have found to be
        wrong solutions. Consider storing wrong predictions instead of wrong goals
    """
    def __init__(self, map):
        self._map_shape = map.shape
        self._visited_areas = np.zeros(self._map_shape, dtype=bool)
        self.world_estimator = WorldEstimatingNetwork()
        self.digit_classifier = DigitClassificationNetwork()

        self._goal = None
        self._wrong_goals = []

        self._delta_direction_map = {
            (-1, 0): 'left',
            (1, 0): 'right',
            (0, -1): 'up',
            (0, 1): 'down'
        }

    def resetNavigator(self):
        self._goal = None
        self._wrong_goals = []
        self._visited_areas = np.zeros(self._map_shape, dtype=bool)

    def _get_neighbors(self, loc, map):
        """
        Get neighbors without corners
        """
        candidate_neighbors = [
            (loc[0] - 1, loc[1]),
            (loc[0] + 1, loc[1]),
            (loc[0], loc[1] - 1),
            (loc[0], loc[1] + 1),
        ]

        return np.array([n for n in candidate_neighbors if 0 <= n[0] < map.shape[0] - 1 and 0 <= n[1] < map.shape[1] - 1 and n[0] != 27 and n[1] != 27])

    def _get_num_unseen_neighbors(self, loc, map):
        neighbors = self._get_neighbors(loc, map)
        return (np.sum(map[neighbors[:, 1], neighbors[:, 0]] == 128))
    
    def _predict_image(self, map, ):
        """
        Outputs softmax of network output
        """
        seen = map != 128

        world_estimation = self.world_estimator.runNetwork(map, seen)
        digit_prediction = self.digit_classifier.runNetwork(world_estimation)
        digit_prediction_softmax = np.exp(digit_prediction - np.max(digit_prediction))
        digit_prediction_softmax /= np.sum(digit_prediction_softmax)

        return digit_prediction_softmax
    
    def _update_goal(self, map, force_goal=False):
        """
        Runs CNNs on current map, if confidence > 90%, set goal corresponding to digit
        """
        predictions = self._predict_image(map)
        if np.max(predictions) > 0.9 or force_goal:
            prediction = np.argmax(predictions)
            
            if prediction in [0, 1, 2]:
                self._goal = (0, 27)
            elif prediction in [3, 4, 5]:
                self._goal = (27, 27)
            else:
                self._goal = (27, 0)

            if self._goal in self._wrong_goals:
                print("wrong goal")
                self._goal = None

    def _check_wrong_goal(self, robot_loc):
        if self._goal in [(0, 27), (27, 27), (27, 0)] and robot_loc == self._goal:
            self._wrong_goals.append(self._goal)
            self._goal = None

    def getAction(self, robot, map):
        robot_loc = robot.getLoc()
        self._visited_areas[robot_loc[1], robot_loc[0]] = 1

        self._check_wrong_goal(robot_loc)

        if self._goal:
            dx = self._goal[0] - robot_loc[0]
            dy = self._goal[1] - robot_loc[1]

        else:
            # Points we have visited will be 0
            unvisited_seen_map = map * (~self._visited_areas & (map != 128))
            
            fully_revealed_digit = np.max(map) > 128 and np.max(unvisited_seen_map) < 128
            self._update_goal(map, force_goal=fully_revealed_digit)

            if np.max(unvisited_seen_map) == 0:
                # Go to the adjacent point with the most unseen neighbors
                neighbors = self._get_neighbors(robot_loc, map)
                
                # TODO: Need a tie breaker, num neighbors to encourage avoiding edges
                max_information_gain_neighbor = max([(self._get_num_unseen_neighbors(n, map), n) for n in neighbors],
                                                key=lambda x: x[0])[1]

                # unvisited_points = np.argwhere(~self._visited_areas)
                # y, x = unvisited_points[np.argmin(np.linalg.norm((unvisited_points - (robot_loc[1], robot_loc[0])), ord=1, axis=1))]

                dx = max_information_gain_neighbor[0] - robot_loc[0]
                dy = max_information_gain_neighbor[1] - robot_loc[1]
            else:
                target = np.argmax(unvisited_seen_map)
                y, x = np.unravel_index(target, map.shape)

                dx = x - robot_loc[0]
                dy = y - robot_loc[1]


        if abs(dx) >= abs(dy):
            delta = (1 * np.sign(dx), 0)
        else:
            delta = (0, 1 * np.sign(dy))


        return self._delta_direction_map[delta]
