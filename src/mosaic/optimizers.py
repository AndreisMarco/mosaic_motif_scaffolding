import equinox as eqx
import jax
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
from typing import Callable
from mosaic.common import TOKENS, is_state_update, has_state_index, LossTerm, LinearCombination
from abc import ABC, abstractmethod

import time
AbstractLoss = LossTerm | LinearCombination

def _print_iter(i, aux):
    # Collect scalar-float leafs and their path
    def is_scalar_float(x):
        return isinstance(x, (float, jax.Array, np.ndarray)) and jnp.ndim(x) == 0
    metrics = {
        jax.tree_util.keystr(k, simple=True, separator='.'): float(v)
        for k, v in jax.tree_util.tree_leaves_with_path(aux)
        if is_scalar_float(v)
        and "state_index" not in jax.tree_util.keystr(k, simple=True, separator=".")
        }
    print(i, " | ".join(f"{k:<5}: {v:>10.2f}" for k, v in metrics.items()))


# Split this up so changing optim parameters doesn't trigger re-compilation of loss function
def _eval_loss_and_grad(
    loss_function: AbstractLoss, x, key, *, serial_evaluation = False, sample_loss = False
):
    """
    Evaluates the loss function and its gradient.

    Args:
    - loss_function: ...
    - x: soft sequence (N x 20 array with each row in the simplex)
    - key: jax random key
    - serial_evaluation: if True, evaluate each loss function in the list sequentially, to save memory
    - sample_loss: if True *and* loss is a LinearCombination, randomly sample one of the loss functions to evaluate with probability proportional to its weight.
    
    Returns:
    - ((value, aux), g): value of the loss function and auxiliary information, and gradient of the loss with respect to x

    """
    assert not (serial_evaluation and sample_loss), "serial_evaluation and sample_loss cannot both be True"

    if sample_loss:
        assert isinstance(loss_function, LinearCombination), "sample_loss can only be used with LinearCombination loss functions"
        w_total = loss_function.weights.sum()
        idx = jax.random.choice(key, len(loss_function.l), p=loss_function.weights / w_total)
        key = jax.random.fold_in(key, 0)
        return _eval_loss_and_grad(loss_function.l[idx], x, key)


    if serial_evaluation:
        assert isinstance(loss_function, LinearCombination), "serial_evaluation can only be used with LinearCombination loss functions"
        results = [
            (w, _eval_loss_and_grad(l, x, jax.random.fold_in(key, idx)))
            for (idx, (w, l)) in enumerate(zip(loss_function.weights, loss_function.l))
        ]
        v = sum(w * r[0][0] for (w, r) in results)
        aux = [r[0][1] for (w, r) in results]
        g = sum(w * r[1] for (w, r) in results)
        return (v, aux), g
       
    # standardize input to avoid recompilation
    x = np.array(x, dtype=np.float32)
    (v, aux), g = _____eval_loss_and_grad(loss_function, x=x, key=key)
    return (jnp.nan_to_num(v, nan = 1000000.0), aux), jnp.nan_to_num(g - g.mean(axis=-1, keepdims=True))

# more underscores == more private
@eqx.filter_jit
def _____eval_loss_and_grad(loss, x, key):
    return eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)

# this function is a mess, but it's used to update stateful loss functions. see comments in mosaic/common.py
# or explanation in mosaic/stateful_loss_explanation.ipynb
def update_states(aux, loss):
    # Collect new_states and the id of their losses
    state_index_to_update = [(x[0].id, x[1])
                             for x in jax.tree.leaves(aux, is_leaf=is_state_update)
                             if is_state_update(x)]
    
    # for multisample losses, as standard we only keep the first new generated state
    state_index_to_update = {
        (int(k.squeeze()) if isinstance(k, np.ndarray) else int(k)): 
        (v[0] if isinstance(k, np.ndarray) else v)
        for k, v in state_index_to_update
        }

    # get loss terms to update
    def get_modules_to_update(loss):
        return tuple([x
                      for x in jax.tree.leaves(loss, is_leaf=has_state_index)
                      if has_state_index(x)])
    # PyTree surgery to update states
    def replace_fn(module):
        return module.update_state(state_index_to_update[int(module.state_index.id)])
    return eqx.tree_at(get_modules_to_update, loss, replace_fn=replace_fn)

from scipy.special import softmax, log_softmax 
def _proposal(sequence, g, temp, alphabet_size: int = 20):
    input = np.eye(alphabet_size)[sequence]
    g_i_x_i = (g * input).sum(-1, keepdims=True)
    logits = -((input * g).sum(-1, keepdims=True) - g_i_x_i + g) / temp
    return softmax(logits, axis=-1), log_softmax(logits, axis=-1)

def gradient_MCMC(
    loss,
    sequence: Int[Array, "N"],
    temp=0.001,
    proposal_temp=0.01,
    max_path_length=2,
    steps=50,
    alphabet_size: int = 20,
    key: None = None,
    detailed_balance: bool = False,
    fix_loss_key: bool = True,
    serial_evaluation: bool = False,
):
    """
    Implements the gradient-assisted MCMC sampler from "Plug & Play Directed Evolution of Proteins with
    Gradient-based Discrete MCMC." Uses first-order taylor approximation of the loss to propose mutations.

        WARNING: Fixes random seed used for loss evaluation.

    Args:
    - loss: log-probability/function to minimize
    - sequence: initial sequence
    - proposal_temp: temperature of the proposal distribution
    - temp: temperature for the loss function
    - max_path_length: maximum number of mutations per step
    - steps: number of optimization steps
    - key: jax random key
    - detailed_balance: whether to maintain detailed balance

    """

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    key_model = key
    (v_0, aux_0), g_0 = _eval_loss_and_grad(
        loss, jax.nn.one_hot(sequence, alphabet_size), key=key_model, serial_evaluation=serial_evaluation
    )
    for iter in range(steps):
        start_time = time.time()
        ### generate a proposal

        for i in range(50):
            proposal = sequence.copy()
            mutations = []
            log_q_forward = 0.0
            path_length = jax.random.randint(
                key=jax.random.key(np.random.randint(10000)),
                minval=1,
                maxval=max_path_length + 1,
                shape=(),
            )
            key = jax.random.fold_in(key, 0)
            for _ in range(path_length):
                p, log_p = _proposal(proposal, g_0, proposal_temp, alphabet_size=alphabet_size)
                mut_idx = jax.random.choice(
                    key=key,
                    a=len(np.ravel(p)),
                    p=np.ravel(p),
                    shape=(),
                )
                key = jax.random.fold_in(key, 0)
                position, AA = np.unravel_index(mut_idx, p.shape)
                log_q_forward += log_p[position, AA]
                mutations += [(position, AA)]
                proposal = proposal.at[position].set(AA)
            # check if proposal is same as current sequence
            if np.all(proposal == sequence):
                print(f"\t {i}: proposal is the same as current sequence, skipping.")
                #_print_iter(iter, {"": aux_0, "time": time.time() - start_time}, v_0)
                #continue
            else:
                break
        muts = ", ".join([f"{pos}:{aa}" for (pos, aa) in mutations])
        print(f"Proposed mutations: {muts}")
        
        ### evaluate the proposal
        (v_1, aux_1), g_1 = _eval_loss_and_grad(
            loss, jax.nn.one_hot(proposal, alphabet_size), key=key_model if fix_loss_key else key, serial_evaluation=serial_evaluation
        )

        # next bit is to calculate the backward probability, which is only used
        # if detailed_balance is True
        prop_backward = proposal.copy()
        log_q_backward = 0.0
        for position, AA in reversed(mutations):
            p, log_p = _proposal(prop_backward, g_1, proposal_temp, alphabet_size=alphabet_size)
            log_q_backward += log_p[position, AA]
            prop_backward = prop_backward.at[position].set(AA)

        log_acceptance_probability = (v_0 - v_1) / temp + (
            (log_q_backward - log_q_forward) if detailed_balance else 0.0
        )

        log_acceptance_probability = min(0.0, log_acceptance_probability)

        print(
            f"iter: {iter}, accept {np.exp(log_acceptance_probability): 0.3f} {v_0: 0.3f} {v_1: 0.3f} {log_q_forward: 0.3f} {log_q_backward: 0.3f}"
        )

        
        print()
        if -jax.random.exponential(key=key) < log_acceptance_probability:
            sequence = proposal
            (v_0, aux_0), g_0 = (v_1, aux_1), g_1
        
        _print_iter(iter, {"": aux_0, "time": time.time() - start_time}, v_0)
        

        key = jax.random.fold_in(key, 0)

    return sequence

class TrajectoryLogger:
    def __init__(self, is_leaf=None):
        self.history = None
        self.is_leaf = is_leaf or \
                            (lambda x: isinstance(x, list) and not isinstance(x[0], dict))

    def _to_cpu(self, x):
        if isinstance(x, (jax.Array, jnp.ndarray)):
            return np.array(x)
        return x

    def update(self, aux):
        # Initialize
        if self.history is None:
            self.history = jax.tree.map(lambda x: [self._to_cpu(x)], 
                                         aux)
        # Update
        else:
            self.history = jax.tree.map(lambda traj, new_value: traj + [self._to_cpu(new_value)], 
                                        self.history, 
                                        aux,
                                        is_leaf=self.is_leaf)
    def stack_arrays(self):
        def _stack(lst):
            try:
                return np.stack(lst)
            except (ValueError, TypeError):
                return lst
        return jax.tree_util.tree_map(_stack, 
                                      self.history, 
                                      is_leaf=self.is_leaf)    

class PSSMOptimizer(ABC):
    def __init__(self,
                loss_fn, 
                n_steps: int = 50,
                max_gradient_norm: float | None = None,
                log_trajectory: bool = False,
                update_loss_state: bool = False,
                serial_evaluation: bool = False,
                sample_loss: bool = False):
        
        # optimization settings
        self.loss_fn = loss_fn
        self.n_steps = n_steps
        self.max_gradient_norm = max_gradient_norm

        # loss evaluation settings
        self.update_loss_state = update_loss_state
        self.serial_evaluation = serial_evaluation
        self.sample_loss = sample_loss

        # logging settings
        self.log_trajectory = log_trajectory
        self.logger = TrajectoryLogger() if log_trajectory else None

    @abstractmethod
    def step(self, state, key):
        """
        Optimizer specifc step 
        """
        pass

    def run(self, 
            pssm_init: Float[Array, "N 20"],
            key):
        
        best_loss = np.inf
        state = {"x": pssm_init}
        best_pssm = pssm_init

        self.max_gradient_norm =  self.max_gradient_norm if self.max_gradient_norm is not None \
                             else np.sqrt(pssm_init.shape[0]) # -_> potentially use a gradient fn from package or implement

        for i in range(self.n_steps):
            start_time = time.time()

            # Optimizer specifc step
            state, loss, aux = self.step(state, key)
            key = jax.random.fold_in(key, 0)

            # Update stateful losses
            if self.update_loss_state:
                self.loss_fn = update_states(aux, self.loss_fn)
            
            aux.update({"optim": {
                "loss": loss, 
                "nnz": (state["x"] > 0.01).sum(-1).mean(), 
                "time": time.time() - start_time
            }})

            if self.log_trajectory:
                self.logger.update(aux)

            # Update best
            if loss < best_loss and not np.isnan(loss):
                best_pssm = state["x"]
            # Log to terminal
            _print_iter(i, aux)
    
        trajectory = self.logger.stack_arrays() if self.log_trajectory else None
        final_pssm = state["x"]
        return final_pssm, best_pssm, trajectory
    
class SimplexAPGM(PSSMOptimizer):
    def __init__(self, stepsize, scale=1, momentum=0.0, **kwargs):
        super().__init__(**kwargs)
        self.stepsize = stepsize
        self.momentum = momentum
        self.scale = scale

    @staticmethod
    def _projection_simplex(V, z=1):
        """
        From https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
        Projection of x onto the simplex, scaled by z:
            P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
        z: float or array
            If array, len(z) must be compatible with V
        """
        V = np.array(V, dtype=np.float64)
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    def step(self, state, key):
        x = self._projection_simplex(state["x"])
        x_prev = state["x_prev"] if "x_prev" in state.keys() else state["x"] 
        v = jax.device_put(x + self.momentum * (x - x_prev))

        (loss, aux), g = _eval_loss_and_grad(
            x=v,
            loss_function=self.loss_fn,
            key=key,
            serial_evaluation=self.serial_evaluation,
            sample_loss=self.sample_loss,
        )

        n = np.sqrt((g**2).sum())
        if n > self.max_gradient_norm:
            g = g * (self.max_gradient_norm / n)
        
        state["x"] = self._projection_simplex(self.scale * (v - self.stepsize * g))
        state["x_prev"] = x

        return state, loss, aux

class LogitAPGM(PSSMOptimizer):
    def __init__(self, stepsize, scale=1.0, momentum=0.0, **kwargs):
        super().__init__(**kwargs)
        self.stepsize = stepsize
        self.momentum = momentum
        self.scale = scale

    def step(self, state, key):
        if "x_logit" not in state: 
            state["x_logit"] = jax.nn.log_softmax(state["x"], axis=-1) 
            state["x_prev_logit"] = state["x_logit"]
        x_logit = state["x_logit"]
        x_prev_logit = state["x_prev_logit"]
        
        v = jax.device_put(x_logit + self.momentum * (x_logit - x_prev_logit))
        (loss, aux), g = _eval_loss_and_grad(
            x=softmax(v, axis=-1),
            loss_function=self.loss_fn,
            key=key,
            serial_evaluation=self.serial_evaluation,
            sample_loss=self.sample_loss,
        )

        n = np.sqrt((g**2).sum())
        if n > self.max_gradient_norm:
            g = g * (self.max_gradient_norm / n)
        
        state["x_logit"] = self.scale * (v - self.stepsize * g) 
        state["x"] = softmax(state["x_logit"], axis=-1)
        state["x_prev_logit"] = x_logit

        return state, loss, aux


