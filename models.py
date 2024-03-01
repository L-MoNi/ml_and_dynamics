##### MODELS ####
# 
# This script defines the NeuralODE and all necessary
# modules, including a simple MLP class.
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

### MODELS ###

class Linear(eqx.Module):

    weights : ArrayLike
    bias    : ArrayLike

    def __init__(self, shape:Tuple[int], *, key:jr.PRNGKey) -> None:
        """Construct a linear layer that implements an affine transformation 
           of the input.

        Input
        shape - the shape of the weight matrix
        key   - key for the pseudo-random number generator   
        """

        # define input and output dimension
        dim_in, dim_out = shape

        # initialize the weights using Kaiming initialization
        self.weights = jr.normal(shape=(dim_in, dim_out), key=key)*np.sqrt(2.0/dim_in)

        # initialize the biases to zero
        self.bias = jnp.zeros(dim_out)


    def __call__(self, x:ArrayLike):
        """Compute the affine transformation of the input.
        
        Input
        x - input value consistent with the input dimension of the Linear layer.
        
        Output
        y - transformed input
        """
        jnp.asarray(x)

        # affine transformation
        return jnp.dot(x, self.weights) + self.bias

class MLP(eqx.Module):

    layers : List
    depth : int
    activation : Callable

    def __init__(self, dims:List, activation:Callable=jax.nn.selu, *, key:jr.PRNGKey) -> None:
        """Construct a multilayer perceptron of arbitrary depth and width.
        
        Input
        dims - list of dimensions s.t. the first entry is the input dimension,
               the last the output dimension, and all other specify hidden dimensions.
        activation - activation function to be applied elementwise after all but the last layer.
        key - key for the pseudo-random number generator
        
        """
        # determine the number of layers
        self.depth = len(dims) - 1

        # get keys for initialization
        init_keys = jr.split(key, self.depth)

        self.layers = []
        for i in range(self.depth):

            self.layers.append(Linear((dims[i],dims[i+1]), key=init_keys[i]))
        
        # set the activation function
        self.activation = activation

    def __call__(self,t:ArrayLike, x:ArrayLike, args:List) -> ArrayLike:
        """Passes one sample through the MLP.
        
        Input
        t - dummy variable for ODE application
        x - input value, needs to be consistent with input dimension of the MLP
        args - dummy variable for ODE application

        Output
        y - sample after passing it through the MLP. There is no non-linearity after the last layer.
        """

        x = jnp.asarray(x)

        for i in range(self.depth - 1):

            x = self.layers[i](x)
            x = self.activation(x)

        x = self.layers[1](x)

        return x

class NeuralODE(eqx.Module):

    vector_field : eqx.Module

    def __init__(self, dims:List, *, key=jr.PRNGKey) -> None:
        """Defines a NeuralODE parametrized by a MLP.
        
        Input
        dims - dimensions of the MLP layers.
        activation - activation after hidden layers of the MLP
        key - key for pseudo-random number generator
        """

        # initialize the MLP that parameterizes the ODEs vector field.
        self.vector_field = MLP(dims, key=key)

    def __call__(self, ts:ArrayLike, y0:ArrayLike) -> ArrayLike:
        """Numerically integrate the ODE using Dormant-Prince solver with
           adaptive step size.
        
        Input
        ts - time points at which to safte the solution
        y0 - the initial condition.
        """

        # Hard coded parameters
        REL_TOL = 1e-3   # relative tolerance
        ABS_TOL = 1e-6   # absolute tolerance
        MAX_STEPS = 8192 # maximum number of iterations


        adaptive_stepsize = dfx.PIDController(rtol=REL_TOL,
                                              atol=ABS_TOL)

        solution = dfx.diffeqsolve(
            dfx.ODETerm(self.vector_field), # vector field
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