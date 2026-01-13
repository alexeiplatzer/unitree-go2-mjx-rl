# Todo list

## Short term

- [X] improve the non-vision teacher student env and test it
- [X] rethink configuration of amount of tiles in colored map terrain
- [X] check optimizer config initialization and usage
- [X] check if the ground plane collides with the tiles in color map terrain
- [X] think of improvements to target-reaching (variance reduction, goal achievement)
- [X] think of improvements to vision students, and vision teachers, improve CNNs, training
- [X] write down configs for everything
- [X] verify that all config-examples function on the GPU server
- [X] check maximum memory utilization for the final, heaviest experiment-config
- [X] save configs when running an experiment 
- [ ] visualize examples from more angles, with GPU also, improve camera angles  
- [X] analyze opportunities for privileged critic
Daily line
- [ ] improve and persist resulting evaluation and convergence data from the plots
- [ ] develop speed benchmarks for different parts

## Mid term

- [ ] add more robust checkpointing to the training
- [ ] adaptable learning rate
- [ ] early termination ?
- [ ] verify the RNN architecture with examples
- [ ] develop a simplified environment for testing the RNN setup
- [ ] try to train with a simple MLP model
- [X] draft a plan for the final paper
- [ ] check how multiple GPUs can be used for vision setups

## Long term

- [ ] update the notebooks, especially the universal one
- [ ] obstacle env has some nan issues, check rewards calcs
- [ ] standardize and make configurable observation-related keywords
- [ ] add obstacle position randomization into the vision env reset
- [ ] add pure tracking and forward moving cumulative reward to metrics, no normalizations.
- [ ] add curriculum training support but to the package
- [ ] add nice post-training rendering and results-saving
- [ ] update the universal notebook, try to add curriculum training there perhaps
- [ ] organize and minimize all the different notebooks on github, kaggle and colab
- [ ] update the notion workspace
- [ ] update the readme
- [ ] add docstrings everywhere
