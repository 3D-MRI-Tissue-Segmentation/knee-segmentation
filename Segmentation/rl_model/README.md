# HiL Rl

## Plan

- Explore the annotation performance increase for manually reannotating segmentations.

## Definitions

- Q learning
- DQN
- Uniform experience replay
- Target model
- Double DQN
- Dueling DQN
- Prioritised experience replay

## Features

- [x] Naive Q Learner
- [x] DQN
- [x] DQN with uniform experience replay
- [x] DQN with uniform experience replay and target model
- [ ] DQN with prioritised expereince replay and target model
- [ ] Dueling DQN with prioritised expereince replay and target model


## Preliminary Results

- Cartpole:
    - Q learning algorthm converges to a solution without any modification of the reward function.
    - DQN required a penalty for failure to be added to the reward function to converge to a solution.

## Test Environments

- [x] Cartpole
- [ ] ITK Snap
 
## Segmentation Engine

- http://www.itksnap.org/pmwiki/pmwiki.php?n=SourceCode.SourceCode

## Resources

- https://www.freecodecamp.org/news/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682/
    - Target network
    - Double DQNs
    - Dueling DQN
    - Prioritised Experience Replay
- https://danieltakeshi.github.io/2019/07/14/per/
- https://arxiv.org/pdf/1812.02648.pdf
- https://arxiv.org/abs/1511.05952
- https://arxiv.org/abs/1905.12726
