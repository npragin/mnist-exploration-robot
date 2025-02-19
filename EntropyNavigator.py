import numpy as np
from networkFolder.functionList import WorldEstimatingNetwork, DigitClassificationNetwork
from copy import deepcopy

class EntropyNavigator:
    """
    Considers the information gain in each of the four directions, goes to most IG
    """
    def __init__(self, map):
        self._map_shape = map.shape
        self._visited_areas = np.zeros(self._map_shape, dtype=bool)
        self.world_estimator = WorldEstimatingNetwork()
        self.digit_classifier = DigitClassificationNetwork()

        self._goal = None
        self._wrong_goals = []
        self._wrong_predictions = []

        self._delta_direction_map = {
            (-1, 0): 'left',
            (1, 0): 'right',
            (0, -1): 'up',
            (0, 1): 'down'
        }

    def resetNavigator(self):
        self._goal = None
        self._wrong_goals = []
        self._wrong_predictions = []
        self._visited_areas = np.zeros(self._map_shape, dtype=bool)

    def _get_neighbors(self, loc, map):
        """
        Get neighbors, excluding corners
        """
        candidate_neighbors = [
            (loc[0] - 1, loc[1]),
            (loc[0] + 1, loc[1]),
            (loc[0], loc[1] - 1),
            (loc[0], loc[1] + 1),
        ]

        return np.array([n for n in candidate_neighbors if 0 <= n[0] < map.shape[0] - 1 and 0 <= n[1] < map.shape[1] - 1 and n[0] != 27 and n[1] != 27])

    def _get_unvisited_neighbors(self, loc, map):
        """
        Get unvisited neighbors, excluding corners
        """
        candidate_neighbors = [
            (loc[0] - 1, loc[1]),
            (loc[0] + 1, loc[1]),
            (loc[0], loc[1] - 1),
            (loc[0], loc[1] + 1),
        ]

        return np.array([n for n in candidate_neighbors if 0 <= n[0] < map.shape[0] - 1 and 0 <= n[1] < map.shape[1] - 1 and n[0] != 27 and n[1] != 27 and self._visited_areas[n[1], n[0]] == 0])
    
    def _get_unseen_neighbors(self, loc, map):
        """
        Get unseen neighbors, excluding corners
        """
        candidate_neighbors = [
            (loc[0] - 1, loc[1]),
            (loc[0] + 1, loc[1]),
            (loc[0], loc[1] - 1),
            (loc[0], loc[1] + 1),
            (loc[0] - 1, loc[1] - 1),
            (loc[0] + 1, loc[1] + 1),
            (loc[0] + 1, loc[1] - 1),
            (loc[0] - 1, loc[1] + 1),
        ]

        return np.array([n for n in candidate_neighbors if 0 <= n[0] < map.shape[0] - 1 and 0 <= n[1] < map.shape[1] - 1 and n[0] != 27 and n[1] != 27 and map[n[1], n[0]] == 128])

    def _predict_image(self, map):
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
        # Set confidence in known wrong predictions to 0
        predictions[0, np.array(self._wrong_predictions, dtype=np.int32)] = 0
        
        if np.max(predictions) > 0.9 or force_goal:
            prediction = np.argmax(predictions)

            if prediction in [0, 1, 2]:
                self._goal = (0, 27)
            elif prediction in [3, 4, 5]:
                self._goal = (27, 27)
            else:
                self._goal = (27, 0)

    def _check_wrong_goal(self, robot_loc):
        if self._goal in [(0, 27), (27, 27), (27, 0)] and robot_loc == self._goal:
            self._wrong_goals.append(self._goal)

            if self._goal == (0, 27):
                self._wrong_predictions.extend([0, 1, 2])
            elif self._goal == (27, 27):
                self._wrong_predictions.extend([3, 4, 5])
            elif self._goal == (27, 0):
                self._wrong_predictions.extend([6, 7, 8, 9])
            self._goal = None

    def _get_entropy(self, map, loc):
        unseen_neighbors = self._get_unseen_neighbors(loc, map)

        if len(unseen_neighbors) == 0:
            return np.inf

        values = [0, 255]
        combinations = np.array(np.meshgrid(*[values for _ in range(len(unseen_neighbors))])).T.reshape(-1, len(unseen_neighbors))
        forecasted_map = deepcopy(map)

        forecasted_entropy_sum = 0
        for i in range(combinations.shape[0]):
            for j in range(combinations.shape[1]):
                forecasted_map[unseen_neighbors[j][1], unseen_neighbors[j][0]] = combinations[i, j]
            
            forecasted_prediction = self._predict_image(forecasted_map)
            forecasted_entropy = -np.sum(forecasted_prediction * np.log(forecasted_prediction))
            forecasted_entropy_sum += forecasted_entropy

        return forecasted_entropy_sum / combinations.shape[0] 

    def getAction(self, robot, map):
        robot_loc = robot.getLoc()
        self._visited_areas[robot_loc[1], robot_loc[0]] = 1

        self._check_wrong_goal(robot_loc)

        if self._goal:
            dx = self._goal[0] - robot_loc[0]
            dy = self._goal[1] - robot_loc[1]

        else:
            unvisited_seen_map = (map + 1) * (~self._visited_areas & (map != 128))
            fully_revealed_digit = np.max(map) > 128 and np.max(unvisited_seen_map) < 129

            self._update_goal(map, force_goal=fully_revealed_digit)

            # TODO: No notion of frontier, we will just chase one frontier indefinitely which becomes a problem as we explore the map more
            #       would be better to check all frontiers instead of the neighbors, but compare the IG/cost instead of just IG
            unvisited_neighbors = self._get_unvisited_neighbors(robot_loc, map)
            if len(unvisited_neighbors):
                # TODO: Consider adding 127 or 129 as a value to use in get_entropy instead of just 0 and 255
                min_entropy_neighbor = min(unvisited_neighbors, key=lambda x: self._get_entropy(map, x))

                dx, dy = (min_entropy_neighbor[0] - robot_loc[0], min_entropy_neighbor[1] - robot_loc[1])
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
