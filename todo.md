# Todo list

## Short term

- [X] check correctness of recurrency handling
- [X] debug dry run on cpu locally all the envs
- [ ] update the notebooks, especially the universal one
- [ ] tryout runs with the notebooks on colab
- [ ] tryout runs on the chair server
- [ ] check server results and plan further action
- [ ] check GPU utilization
- [ ] check how multiple GPUs can be used for vision setups
- [ ] implement visualisation rollouts for recurrent support

## Mid term

- [ ] verify the RNN architecture with examples
- [ ] develop a simplified environment for testing the RNN setup
- [ ] improve and persist resulting evaluation and convergence data from the plots
- [ ] try to train with a simple MLP model
- [ ] draft a plan for the final paper

## Long term

- [ ] obstacle env has some nan issues, check rewards calcs
- [ ] standardize and make configurable observation-related keywords
- [ ] add obstacle position randomization into the vision env reset
- [ ] improve the non-vision teacher student env and test it
- [ ] add pure tracking and forward moving cumulative reward to metrics, no normalizations.
- [ ] add more robust checkpointing to the training
- [X] ~~add an "implements" decorator for checking protocol adherence on definition~~
- [X] ~~configure a curriculum training kaggle pipeline with terrain~~
- [ ] add curriculum training support but to the package
- [ ] add nice post-training rendering and results-saving
- [ ] update the universal notebook, try to add curriculum training there perhaps
- [ ] organize and minimize all the different notebooks on github, kaggle and colab
- [ ] update the notion workspace
- [ ] update the readme
- [ ] add docstrings everywhere
