# Reinforcement Learning for Quadruped Robots Locomotion (Unitree Go2 and others) in Mujoco XLA (MJX)

This package implements Locomotion policies for Quadruped Robots, such as Unitree Go2 using
deep Reinforcement Learning.

## Overview

The package uses Mujoco XLA (MJX) for simulation, Brax for the Reinforcement Learning Pipeline,
and Madrona MJX for rendering during simulation.

To learn quadruped locomotion, a policy neural network is trained with PPO 
(Proximal policy optimization) and a Teacher-Student setup for inferring privileged state
information from visual input.

## Installation

The package can be installed after cloning as an editable package with
`pip install -e .`
