##### DATA HANDLERS ####
# 
# This script defines generators and data loaders for training
# NeuralODEs on time series and classification tasks.
# 
# This code has been inspired by Patrick Kridgers tutorial
# on NeuralODEs in Diffrax. 
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

### FOR TRAINING ON TRAJECTORIES ###

def generate_trajectories(n_trajectories:int, n_points:int, tmax:int, *, key:jr.PRNGKey) -> Tuple[ArrayLike]:
    """Construct a dataset consisting of discrete time points and trajectories
       starting from random initial conditions.

        Input
        n_trajectories - the number of trajectories
        n_points - the number of points saved on the trajectory
        tmax - the endoint in real time
        key - key for the  pseudo-random number generator

        Returns
        (ts, ys) - the dataset with solutions ys of shape (n_trajectories, n_points, n_dims) and 
                   timepoints ts of shape.
    """

    # Local Constants
    REL_TOL = 1e-3   # relative tolerance
    ABS_TOL = 1e-6   # absolute tolerance
    MAX_STEPS = 8192 # maximum number of iterations
    Y0MIN = -0.5     # minimum initial condition possible
    Y0MAX = 0.5      # maximum initial condition possible
    OMEGA = 1        # natural frequency
    ZETA = 0.15      # damping
    NDIMS = 2        # number of dimensions
    TMIN = 0         # first timepoint


    def _vector_field(t:ArrayLike, y:ArrayLike, args:List) -> ArrayLike:

        return jnp.stack([y[1], -ZETA*OMEGA*y[1] - OMEGA**2 * y[0]], axis=-1)


    def _solve(ts:ArrayLike, *, key:jr.PRNGKey) -> ArrayLike:

        # initialize the stepsize adaption
        adaptive_stepsize = dfx.PIDController(rtol=REL_TOL,
                                              atol=ABS_TOL)

        # generate the initial condition uniformly at random
        y0 = jr.uniform(key,
                        shape=(NDIMS,),
                        minval=Y0MIN,
                        maxval=Y0MAX)

        # solve the ODE numerically
        solution = dfx.diffeqsolve(
            dfx.ODETerm(_vector_field), # vector field
            dfx.Dopri5(),                   # solver: Dormand Prince 5(4)
            t0=ts[0],                       # start solving at t0
            t1=ts[-1],                      # stop solving at t1
            dt0=ts[1] - ts[0],              # first choice of time step
            y0=y0,                          # initial conditions
            max_steps=MAX_STEPS,            # maximum number of iterations        
            saveat=dfx.SaveAt(ts=ts),       # return the solution at these timepoints
            stepsize_controller=adaptive_stepsize,  # adaptive stepsize
        )

        return solution.ys

    # generate time steps
    ts = jnp.linspace(TMIN, tmax, n_points)

    keys = jr.split(key, n_trajectories)
    ys = jax.vmap(lambda keys: _solve(ts, key=keys))(keys)

    return ts, ys.reshape((n_trajectories, n_points, NDIMS))

def trajectory_loader(ys:ArrayLike, batch_size:int, * , key:jr.PRNGKey) -> ArrayLike:
    """Creates a generator to loop through the data and returning randomly sampled batches.
       This version is suitable for timeseries tasks.
    
    Input
    ys - the trajectory data
    batch_size - the size of the batches
    key - key for pseudo-random number generation

    Output
    ybatch - one randomly selected batch
    """

    n_trajectories = ys.shape[0]        # the total number of trajectories
    index = jnp.arange(n_trajectories)  # inex into the trajectories

    # makes this a (stochastic) generator that can be repeatedly cycled through  
    while True:

        # shuffle the index of the dataset
        (key, _) = jr.split(key, 2)
        permutation = jr.permutation(key, index)

        batch_start = 0
        batch_end = batch_size

        while batch_start < n_trajectories:

            # select batch_size trajectories uniformly at random
            selected = permutation[batch_start:batch_end]
            yield ys[selected]

            # update counters
            batch_start += batch_size
            batch_end += batch_size


### FOR TRAINING ON CLASSIFICATION ###

def classification_loader(ys:ArrayLike, labels:ArrayLike, batch_size:int, * , key:jr.PRNGKey) -> ArrayLike:
    """Creates a generator to loop through the data and returning randomly sampled batches.
       This version is suitable for classification tasks.
    
    Input
    ys - the data points
    labels - the labels of these points
    batch_size - the size of the batches
    key - key for pseudo-random number generation

    Output
    batch - one randomly selected batch as a tuple of the points and labels.
    """

    assert ys.shape[0] == labels.shape[0]

    n_samples = ys.shape[0]             # the total number of samples in the dataset
    index = jnp.arange(n_samples)       # inex into the the samples and labels

    # makes this a (stochastic) generator that can be repeatedly cycled through  
    while True:

        # shuffle the index of the dataset
        (key, _) = jr.split(key, 2)
        permutation = jr.permutation(key, index)

        batch_start = 0
        batch_end = batch_size

        while batch_start < n_samples:

            # select batch_size trajectories uniformly at random
            selected = permutation[batch_start:batch_end]
            yield (ys[selected], labels[selected])

            # update counters
            batch_start += batch_size
            batch_end += batch_size