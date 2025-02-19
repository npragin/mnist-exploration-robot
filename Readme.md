# Informative Path Planning over MNIST Digits

This project implements navigation strategies for autonomous exploration of MNIST digit environments, combining classical robotics approaches with deep learning. The system demonstrates advanced exploration techniques using information theory and neural networks to efficiently navigate partially observable environments.

<div align="center">
<table>
<tr>
<td align="center" width="50%">
    <img src="assets/example_traj_greedy.gif" alt="Greedy exploration visualization" width="50%"/><br>
    <b>Greedy Navigation Strategy</b>
</td>
<td align="center" width="50%">
    <img src="assets/example_traj_entropy.gif" alt="Entropy-based exploration visualization" width="50%"/><br>
    <b>Entropy-Based Navigation Strategy</b>
</td>
</tr>
</table>
</div>

## Project Overview
### Environment
- **Grid World**: 28x28 binary grid representing an MNIST handwritten digit
- **Partial Observability**: Robot can only observe its 8-connected neighborhood
- **Dynamic Goals**: Three possible goal locations based on digit classification:
  - Bottom-left (0,27) for digits 0-2
  - Bottom-right (27,27) for digits 3-5
  - Top-right (27,0) for digits 6-9

### Key Challenges
1. **Partial Observability**: The robot must make decisions with incomplete information
2. **Exploration-Exploitation Tradeoff**: Balancing between exploring to identify the digit and exploiting current knowledge to reach the goal
3. **Uncertainty Management**: Handling potential misclassifications and wrong goal attempts

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
1. Calculates the expected information gain (reduction in entropy) for exploring each neighboring cell
2. Moves toward the cell that maximizes expected information gain
3. When the digit is fully revealed or classification confidence exceeds 90%, navigates to the goal corresponding to the predicted digit

Key features:
- Considers all possible binary (0/255) value combinations for unseen neighbors
- Maintains a probabilistic belief state about the digit identity
- Uses entropy reduction as the primary exploration metric

### 2. Greedy Navigator
A computationally efficient, myopic exploration strategy:
- Moves to the brightest unvisited point in the map
  - Tie breaker: Moves to the neighboring cell with the most unseen neighbors
- When the digit is fully revealed or classification confidence exceeds 90%, navigates to the goal corresponding to the predicted digit

## Performance Analysis

### Scoring Mechanism
- **Success Reward**: +100 points for reaching correct goal
- **Movement Penalty**: -1 point per move
- **Mistake Penalty**: -400 points for reaching wrong goal

### Strategy Comparison

| Strategy | Avg Score | Avg Steps | Computation Time |
|----------|-----------|-----------|-----------------|
| Entropy  | -216.68   | 237.88    | 16.38s         |
| Greedy   | -271.90   | 262.14    | 1.75s          |

The performance metrics above are based on 100 trials for each strategy on the same maps:
- Entropy Navigator achieves ~20% better average scores
- Entropy Navigator requires ~9% fewer steps on average
- Entropy is ~9.4x slower but justifies cost with better performance

3. **Computational Cost**
   - Entropy: ~16.38s per trial (O(2^n) complexity)
   - Greedy: ~1.75s per trial (O(1) complexity)

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
- Entropy Navigator's frontier exploration could be improved by considering all frontiers instead of just neighbors
- Greedy Navigator may get stuck in loops if the digit prediction is wrong and fully revealed

## Citations
- Original Challenge Design: [Jeff Caley (@caleytown)](https://github.com/caleytown)
- [MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/)
