# Todo list

## Short term
- [X] refactor the color rand function into the package
- [X] add native support for madrona rendering visualization to the package
- [X] add the ability to add custom cameras to robots
- [ ] add a more varied color palette to color randomization
- [ ] normalize the position and location of grid tiles in color rand envs
- [X] recustomize the strings used for defining vision and teacher-student architectures
- [ ] figure out how to pass privileged terrain info into the environment
- [X] add square size customization support

## Mid term
- [ ] finish setting up RNN training
- [ ] develop a simplified environment for testing the RNN setup
- [ ] try to set up the department's GPUs
- [ ] try to train with a simple MLP model
- [ ] improve and persist resulting evaluation and convergence data from the plots
- [ ] draft a plan for the final paper
- [ ] standardize and make configurable observation-related keywords
- [ ] add a terrain generation config base class and corresponding config classes
- [ ] check how multiple GPUs can be used for vision setups

## Long term

### Package
- [ ] add obstacle position randomization into the vision env reset
- [ ] improve the non-vision teacher student env and test it
- [ ] add pure tracking and forward moving cumulative reward to metrics, no normalizations.
- [ ] add more robust checkpointing to the training
- [ ] add an "implements" decorator for checking protocol adherence on definition

### Notebooks
- [ ] configure a curriculum training kaggle pipeline with terrain
- [ ] add nice post-training rendering and results-saving
- [ ] update the universal notebook, try to add curriculum training there perhaps
- [ ] organize and minimize all the different notebooks on github, kaggle and colab

### Documentation
- [ ] update the notion workspace
- [ ] update the readme
- [ ] add docstrings everywhere