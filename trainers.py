##### TRAINERS ####
# 
# This script defines the functions necessary to train
# NeuralODEs on timeseries and classification tasks.
#
# This code has been inspired by Patrick Kridgers tutorial
# on NeuralODEs in Diffrax and the UVA Deep Learning Tutorials.
#   
#  Moritz Laber (last update: 2024/03/01)

### IMPORT ###
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.typing import ArrayLike
import equinox as eqx
import optax as tx
import diffrax as dfx
import numpy as np
from typing import List, Tuple, Callable
from tqdm import tqdm

from models import *
from data_handlers import *


### TRAINING ON TRAJECTORIES ###
def trajectory_train(model:eqx.Module, dataset:Tuple[ArrayLike], schedule: List[Tuple], batch_size:int, *, key:jr.PRNGKey) -> eqx.Module:
    """Trains a NeuralODE model on the trajectories in data under the a specific training schedule.
    
    Input
    model - a NeuralODE model
    dataset - tuple of time points and trajectories sampled at these timepoints
    schedule - list of tuples of the form (number of steps, (start time, end time), learning rate)
    batch_size - number of trajectories per batch
    key - key for pseudo-random number generation   

    Output
    model - the trained NeuralODE
    loss_list - the loss after each iteration of training.
    """

    @eqx.filter_value_and_grad
    def loss_val_grad(model, t, y):

        # solve the ode with current parameters and vmap over initial conditions
        ypred = jax.vmap(model, in_axes=(None,0))(t, y[:, 0])

        # calculate the mean square error
        loss_val = jnp.mean(jnp.power(y - ypred, 2))

        return loss_val
    
    @eqx.filter_jit
    def step(model, t, y, state):

        # calculate gradients and loss at the current step
        loss_val, grads = loss_val_grad(model, t, y)

        # calculate the parameter updates
        updates, state = optimizer.update(grads, state)

        # apply the updates to the model
        model = eqx.apply_updates(model, updates)

        return model, loss_val, state
    
    # unpack the dataset
    tdata, ydata = dataset

    # training loop
    loss_list = []
    for n_steps, trange, lr in schedule:

        # initialize the optimizer
        optimizer = tx.adam(learning_rate=lr)
        state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        # truncate the trajectories in time
        ts = tdata[trange[0]:trange[1]]
        ys = ydata[:,trange[0]:trange[1],:]

        for s, ybatch in tqdm(zip(range(n_steps), trajectory_loader(ys, batch_size, key=key))):

            # take a step in the optimization process and store the loss
            model, loss_val, state = step(model, ts, ybatch, state)
            loss_list.append(loss_val.item())
    
    return model, loss_list

### TRAINING FOR CLASSIFICATION ###
def classifier_train(model:eqx.Module, dataset:Tuple[ArrayLike], schedule: List[Tuple], batch_size:int, *, key:jr.PRNGKey) -> eqx.Module:
    """Trains a NeuralODE model for classification under a specific training schedule.
    
    Input
    model - a NeuralODE model
    dataset - tuple of timepoints, initial points  (samples) and their labels
    schedule - list of tuples of the form (number of steps, learning rate)
    batch_size - number of points per batch
    key - key for pseudo-random number generation   

    Output
    model - the trained NeuralODE
    loss_list - the loss after each iteration of training.
    """

    @eqx.filter_value_and_grad
    def loss_val_grad(model, t, y, labels):

        # solve the ode starting at the dataset points
        ypred = jax.vmap(model, in_axes=(None,0))(t, y)

        # use non-learnable projection on the y axis 
        # TODO: integrate learnable projection into the model.
        logits = ypred[:, -1, -1]

        # calculate the binary cross-entropy loss 
        loss_val = jnp.mean(tx.sigmoid_binary_cross_entropy(logits=logits, labels=labels))

        return loss_val
    
    @eqx.filter_jit
    def step(model, t, y, labels, state):

        # calculate gradients and loss at the current step
        loss_val, grads = loss_val_grad(model, t, y, labels)

        # calculate the parameter updates
        updates, state = optimizer.update(grads, state)

        # apply the updates to the model
        model = eqx.apply_updates(model, updates)

        return model, loss_val, state
    
    # unpack the dataset
    ts, ys, labels = dataset

    # training loop
    loss_list = []
    for n_steps, lr in schedule:

        # initialize the optimizer
        optimizer = tx.adam(learning_rate=lr)
        state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        for _, batch in tqdm(zip(range(n_steps), classification_loader(ys, labels, batch_size, key=key))):
            
            # unpack batch
            y_batch, label_batch = batch

            # take a step in the optimization process and store the loss
            model, loss_val, state = step(model, ts, y_batch, label_batch, state)
            loss_list.append(loss_val)

    
    return model, loss_list