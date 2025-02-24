# Informative Path Planning over MNIST Digits

This project implements navigation strategies for autonomous exploration of MNIST digit environments with dynamic goals, combining classical robotics approaches with deep learning.

<div align="center">
<table>
<tr>
<td align="center" width="50%">
    <img src="assets/greedy_trajectories.gif" alt="Greedy exploration visualization" width="50%"/><br>
    <b>Greedy Navigation Strategy</b>
</td>
<td align="center" width="50%">
    <img src="assets/entropy_trajectories.gif" alt="Entropy-based exploration visualization" width="50%"/><br>
    <b>Information-Theoretic Navigation Strategy</b>
</td>
</tr>
</table>
</div>

## Project Overview
**The core goal is for a robot to efficiently navigate to the correct grid corner by identifying a hidden MNIST digit.**
### Environment
- **Grid World**: 28x28 binary grid representing an MNIST handwritten digit
- **Partial Observability**: Robot can only observe its 8-connected neighborhood

### Dynamic Goals
- Three possible goal locations based on digit classification:
  - Bottom-left (0,27) for digits 0-2
  - Bottom-right (27,27) for digits 3-5
  - Top-right (27,0) for digits 6-9
 
### Scoring Mechanism
- **Success Reward**: +100 points for reaching correct goal
- **Movement Penalty**: -1 point per move
- **Mistake Penalty**: -400 points for reaching wrong goal

### Key Challenges
- **Partial Observability**: The robot must make decisions with incomplete information
- **Exploration-Exploitation Tradeoff**: Balancing between exploring to identify the digit and exploiting current knowledge to reach the goal
- **Uncertainty Management**: Handling potential misclassifications and wrong goal attempts

## Navigation Strategies

### 1. Information-Theoretic Navigator (EntropyNavigator)
An advanced exploration strategy using information theory principles:

```python
def _get_entropy(self, map, loc):
    unseen_neighbors = self._get_unseen_neighbors(loc, map)
    combinations = np.array(np.meshgrid(*[values for _ in range(len(unseen_neighbors))]))
    
    forecasted_entropy_sum = 0
    for combination in combinations:
        forecasted_map = self._apply_combination(map, unseen_neighbors, combination)
        prediction = self._predict_image(forecasted_map)
        forecasted_entropy = -np.sum(prediction * np.log(prediction))
        forecasted_entropy_sum += forecasted_entropy
```
1. Calculates the expected information gain (reduction in entropy) for exploring each unvisited neighboring cell
   - Considers all possible binary value (0/255) combinations for unseen neighbors of each candidate cell
2. Moves toward the cell that maximizes expected information gain
3. When the digit is fully revealed or classification confidence exceeds 90%, navigates to the goal corresponding to the predicted digit

Key features:
- Limits computational burden by considering binary combinations
- Maintains a probabilistic belief state about the digit identity
- Uses entropy reduction as the primary exploration metric

### 2. Greedy Navigator
A computationally efficient, myopic exploration strategy:
- Moves to the brightest unvisited point in the map
  - Tie breaker: Moves to the neighboring cell with the most unseen neighbors
- When the digit is fully revealed or classification confidence exceeds 90%, navigates to the goal corresponding to the predicted digit

## Performance Analysis

### Strategy Comparison

The following performance metrics are based on 100 trials across the same maps:

| Strategy | Avg Score | Avg Steps | Computation Time |
|----------|-----------|-----------|-----------------|
| Entropy  | -216.68   | 237.88    | 16.38s         |
| Greedy   | -271.90   | 262.14    | 1.75s          |

- Entropy is ~9.4x slower but justifies cost with better performance

 **Computational Cost**
   - Entropy: O(2^n) complexity in the number of neighbors
   - Greedy: O(1) complexity

## Neural Network Integration

The system utilizes two specialized neural networks:

### 1. World Estimation Network
- **Architecture**: Modified U-Net with partial convolutions
- **Input**: Partially observed grid with binary mask
- **Output**: Complete digit reconstruction

### 2. Digit Classification Network
- **Architecture**: CNN
- **Input**: Reconstructed digit from World Estimation Network
- **Output**: Digit class probabilities

## Future Improvements
- Entropy Navigator could be improved by considering all frontiers instead of just neighbors
- Greedy Navigator may get stuck in loops if the digit prediction is wrong and the digit is fully revealed

## Citations
- Original Challenge Design: [Jeff Caley (@caleytown)](https://github.com/caleytown)
- [MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/)
