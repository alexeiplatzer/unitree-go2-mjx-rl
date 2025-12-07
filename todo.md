# Todo list

## Short term
- [X] adapt training utils and wrap for training for new domain rand architectures
- [X] add a more varied color palette to color randomization
- [X] normalize the position and location of grid tiles in color rand envs
- [X] figure out how to pass privileged terrain info into the environment
- [X] figure out how to merge the FFN and RNN approach with minimal duplication
- [X] implement training fronted for recurrent support
- [X] implement correct evaluation for recurrent support
- [X] finish setting up RNN training
- [ ] update randomization functions interfacing with other subpackages
- [ ] verify the RNN architecture with examples
- [ ] develop a simplified environment for testing the RNN setup
- [ ] try out a very simple training procedure on chair GPUs
- [ ] implement visualisation rollouts for recurrent support
- 
## Mid term
- [ ] check how multiple GPUs can be used for vision setups
- [ ] improve and persist resulting evaluation and convergence data from the plots
- [ ] try to set up the department's GPUs
- [ ] try to train with a simple MLP model
- [ ] draft a plan for the final paper
- [ ] standardize and make configurable observation-related keywords
- [ ] add a terrain generation config base class and corresponding config classes


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