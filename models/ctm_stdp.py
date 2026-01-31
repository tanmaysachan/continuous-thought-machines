"""
STDP CTM: Continuous Thought Machine with Spike-Timing Dependent Plasticity

Implements STDP (Spike-Timing Dependent Plasticity) - a causal variant of Hebbian learning
where connections strengthen based on temporal ordering.

Key difference from symmetric Hebbian:
- Hebbian: ΔW_ij = η * act_i(t) * act_j(t)  (symmetric, simultaneous)
- STDP: ΔW_ij = η * pre_i(t-1) * post_j(t)  (asymmetric, causal)

The causal nature means that if neuron i fires at t-1 and neuron j fires at t,
the connection W_ij strengthens (LTP - Long-Term Potentiation). This captures
the "neurons that fire in sequence wire together" principle from neuroscience.
"""

import torch
import torch.nn as nn
import math


class STDPLateralModule(nn.Module):
    """
    STDP (Spike-Timing Dependent Plasticity) lateral connection module.

    Unlike standard Hebbian learning that uses simultaneous co-activation,
    STDP uses temporal causality: connections strengthen when pre-synaptic
    activity at t-1 precedes post-synaptic activity at t.

    Args:
        d_model: Number of neurons
        stdp_lr: Learning rate for STDP updates (LTP strength)
        stdp_decay: Decay factor to prevent unbounded growth
        lateral_strength: Scaling factor for lateral input contribution
    """
    def __init__(
        self,
        d_model,
        stdp_lr=0.0001,
        stdp_decay=0.999,
        lateral_strength=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.stdp_lr = stdp_lr
        self.stdp_decay = stdp_decay
        self.lateral_strength = lateral_strength

        # STDP weights: accumulated causal co-activation patterns
        # Using None for lazy initialization to avoid MPS buffer issues
        self._stdp_weights = None

        # Buffer for t-1 activations (for causal computation)
        self._prev_activation = None

        # Learnable gate: controls how much lateral input affects each neuron
        self.lateral_gate_weights = nn.Parameter(torch.randn(d_model) * 0.1)

        # Optional: learnable base lateral weights (sparse initialization)
        self.register_parameter(
            'base_lateral',
            nn.Parameter(torch.zeros(d_model, d_model))
        )

        # Pre-compute diagonal mask (1s everywhere except diagonal)
        self.register_buffer('_diag_mask', 1.0 - torch.eye(d_model))

    @property
    def stdp_weights(self):
        """Lazy initialization of STDP weights on first access."""
        if self._stdp_weights is None:
            self._stdp_weights = torch.zeros(
                self.d_model, self.d_model,
                device=self.base_lateral.device,
                dtype=self.base_lateral.dtype
            )
        return self._stdp_weights

    @stdp_weights.setter
    def stdp_weights(self, value):
        self._stdp_weights = value

    def get_effective_weights(self):
        """
        Combine base (learned) weights with STDP (accumulated) weights.
        """
        # Normalize STDP weights to prevent explosion
        stdp_normalized = self.stdp_weights
        max_val = self.stdp_weights.abs().max()
        if max_val > 0:
            stdp_normalized = self.stdp_weights / (max_val + 1e-8)

        effective = self.base_lateral + stdp_normalized

        # Zero diagonal (no self-connections)
        effective = effective * self._diag_mask

        return effective

    def compute_lateral_input(self, activated_state):
        """
        Compute lateral input for each neuron based on other neurons' activations.

        Args:
            activated_state: (B, D) current neuron activations

        Returns:
            lateral_input: (B, D) weighted sum of other neurons' activations
        """
        effective_weights = self.get_effective_weights()  # (D, D)
        # W[i,j] = connection from neuron i to neuron j
        # lateral[j] = sum_i(act[i] * W[i,j]) = influence flowing INTO neuron j
        lateral_input = torch.matmul(activated_state, effective_weights)  # (B, D)

        # Apply per-neuron learnable gate
        gate = torch.sigmoid(self.lateral_gate_weights)  # (D,)
        gated_lateral = gate * lateral_input * self.lateral_strength
        return gated_lateral

    def stdp_update(self, activated_state):
        """
        Update STDP weights based on causal (temporal) co-activation.

        STDP rule (LTP only): ΔW_ij = η * pre_i(t-1) * post_j(t)
        This strengthens connections when neuron i fires before neuron j.

        The resulting weight matrix is asymmetric: W_ij ≠ W_ji in general,
        capturing the directionality of causal influence.

        Args:
            activated_state: (B, D) current neuron activations (post at t)
        """
        if not self.training:
            return

        with torch.no_grad():
            # Mean activation across batch for stability
            current_act = activated_state.mean(dim=0)  # (D,) - post at t

            # Handle first timestep (no t-1 exists)
            if self._prev_activation is None:
                self._prev_activation = current_act.clone()
                return

            prev_act = self._prev_activation  # pre at t-1

            # LTP: pre(t-1) → post(t) strengthens connection
            # W_ij strengthens when neuron i fires at t-1 before neuron j fires at t
            # Shape: prev_act (D,) -> (D, 1), current_act (D,) -> (1, D)
            # Result: (D, D) where [i, j] = pre_i(t-1) * post_j(t)
            delta = self.stdp_lr * (prev_act.unsqueeze(1) @ current_act.unsqueeze(0))

            # Apply decay to old weights and add new delta
            new_weights = self.stdp_weights * self.stdp_decay + delta

            # Zero diagonal (no self-connections)
            self._stdp_weights = new_weights * self._diag_mask

            # Save current activation for next timestep
            self._prev_activation = current_act.clone()

    def reset_activation_buffer(self):
        """Reset t-1 buffer between episodes/batches."""
        self._prev_activation = None

    def reset_stdp(self):
        """Reset STDP weights to zero (useful for new training runs)."""
        if self._stdp_weights is not None:
            self._stdp_weights = torch.zeros_like(self._stdp_weights)
        self._prev_activation = None

    def get_stdp_stats(self):
        """Return statistics about STDP weights for monitoring."""
        with torch.no_grad():
            weights = self.stdp_weights
            # Compute asymmetry: how different W is from W.T
            asymmetry = (weights - weights.T).abs()
            return {
                'mean': weights.mean().item(),
                'std': weights.std().item(),
                'max': weights.max().item(),
                'min': weights.min().item(),
                'sparsity': (weights.abs() < 1e-6).float().mean().item(),
                'top_10_mean': weights.abs().flatten().topk(10).values.mean().item(),
                'asymmetry_mean': asymmetry.mean().item(),
                'asymmetry_max': asymmetry.max().item(),
            }


class STDPCTM(nn.Module):
    """
    CTM with STDP (Spike-Timing Dependent Plasticity) Lateral Connections.

    Wraps an existing CTM and adds:
    1. STDP lateral connection module with temporal tracking
    2. Integration of lateral input into the forward pass
    3. Causal weight updates based on pre(t-1) -> post(t) activity

    The key difference from HebbianCTM is the causal nature of learning:
    - HebbianCTM: neurons that fire together wire together (symmetric)
    - STDPCTM: neurons that fire in sequence wire together (asymmetric)

    Args:
        base_ctm: An existing ContinuousThoughtMachine instance
        stdp_lr: Learning rate for STDP updates
        stdp_decay: Decay factor for STDP weights
        lateral_strength: How strongly lateral input affects neurons
        lateral_injection: Where to inject lateral input ('pre_synapse', 'post_nlm', 'both')
    """
    def __init__(
        self,
        base_ctm,
        stdp_lr=0.0001,
        stdp_decay=0.999,
        lateral_strength=0.1,
        lateral_injection='pre_synapse',
    ):
        super().__init__()

        self.ctm = base_ctm
        self.lateral_injection = lateral_injection

        # Create STDP lateral module
        self.stdp = STDPLateralModule(
            d_model=base_ctm.d_model,
            stdp_lr=stdp_lr,
            stdp_decay=stdp_decay,
            lateral_strength=lateral_strength,
        )

    def forward(self, x, track=False):
        """
        Forward pass with STDP lateral connections.

        Same interface as base CTM, but with causal lateral connections active.
        The activation buffer is reset at the start of each forward pass (new batch).
        """
        B = x.size(0)
        device = x.device

        # Reset t-1 buffer at start of new batch
        self.stdp.reset_activation_buffer()

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []
        stdp_stats_tracking = []

        # --- Featurise Input Data ---
        kv = self.ctm.compute_features(x)

        # --- Initialise Recurrent State ---
        state_trace = self.ctm.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.ctm.start_activated_state.unsqueeze(0).expand(B, -1)

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.ctm.out_dims, self.ctm.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.ctm.iterations, device=device, dtype=torch.float32)

        # --- Initialise Recurrent Synch Values ---
        decay_alpha_action, decay_beta_action = None, None
        self.ctm.decay_params_action.data = torch.clamp(self.ctm.decay_params_action, 0, 15)
        self.ctm.decay_params_out.data = torch.clamp(self.ctm.decay_params_out, 0, 15)
        r_action = torch.exp(-self.ctm.decay_params_action).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-self.ctm.decay_params_out).unsqueeze(0).repeat(B, 1)

        _, decay_alpha_out, decay_beta_out = self.ctm.compute_synchronisation(
            activated_state, None, None, r_out, synch_type='out'
        )

        # --- Recurrent Loop ---
        for stepi in range(self.ctm.iterations):

            # --- Calculate Synchronisation for Input Data Interaction ---
            synchronisation_action, decay_alpha_action, decay_beta_action = \
                self.ctm.compute_synchronisation(
                    activated_state, decay_alpha_action, decay_beta_action,
                    r_action, synch_type='action'
                )

            # --- Interact with Data via Attention ---
            q = self.ctm.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.ctm.attention(
                q, kv, kv, average_attn_weights=False, need_weights=True
            )
            attn_out = attn_out.squeeze(1)

            # === STDP LATERAL INPUT (pre-synapse injection) ===
            if self.lateral_injection in ('pre_synapse', 'both'):
                lateral_input = self.stdp.compute_lateral_input(activated_state)
                activated_state_for_synapse = activated_state + lateral_input
            else:
                activated_state_for_synapse = activated_state

            pre_synapse_input = torch.cat((attn_out, activated_state_for_synapse), dim=-1)

            # --- Apply Synapses ---
            state = self.ctm.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply Neuron-Level Models ---
            activated_state = self.ctm.trace_processor(state_trace)

            # === STDP LATERAL INPUT (post-NLM injection) ===
            if self.lateral_injection in ('post_nlm', 'both'):
                lateral_input = self.stdp.compute_lateral_input(activated_state)
                activated_state = activated_state + lateral_input

            # === STDP UPDATE ===
            # Update lateral weights based on causal relationship: pre(t-1) -> post(t)
            # Uses internally stored t-1 activation
            self.stdp.stdp_update(activated_state)

            # --- Calculate Synchronisation for Output Predictions ---
            synchronisation_out, decay_alpha_out, decay_beta_out = \
                self.ctm.compute_synchronisation(
                    activated_state, decay_alpha_out, decay_beta_out,
                    r_out, synch_type='out'
                )

            # --- Get Predictions and Certainties ---
            current_prediction = self.ctm.output_projector(synchronisation_out)
            current_certainty = self.ctm.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:, :, -1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())
                stdp_stats_tracking.append(self.stdp.get_stdp_stats())

        # --- Return Values ---
        if track:
            import numpy as np
            return (
                predictions,
                certainties,
                (np.array(synch_out_tracking), np.array(synch_action_tracking)),
                np.array(pre_activations_tracking),
                np.array(post_activations_tracking),
                np.array(attention_tracking),
            )
        return predictions, certainties, synchronisation_out

    def reset_stdp(self):
        """Reset STDP weights and activation buffer."""
        self.stdp.reset_stdp()

    def reset_activation_buffer(self):
        """Reset only the t-1 activation buffer (between batches)."""
        self.stdp.reset_activation_buffer()

    def get_stdp_weights(self):
        """Get current STDP weight matrix for visualization."""
        return self.stdp.stdp_weights.detach().cpu()

    def get_effective_lateral_weights(self):
        """Get combined base + STDP weights."""
        return self.stdp.get_effective_weights().detach().cpu()

    def get_asymmetry_matrix(self):
        """Get the asymmetry matrix (W - W.T) showing causal directionality."""
        weights = self.stdp.stdp_weights.detach().cpu()
        return weights - weights.T


def wrap_ctm_with_stdp(
    ctm,
    stdp_lr=0.0001,
    stdp_decay=0.999,
    lateral_strength=0.1,
    lateral_injection='pre_synapse',
):
    """
    Convenience function to wrap an existing CTM with STDP lateral connections.

    Args:
        ctm: Existing ContinuousThoughtMachine instance
        stdp_lr: Learning rate for STDP updates (causal potentiation strength)
        stdp_decay: Decay rate (closer to 1 = longer memory)
        lateral_strength: How much lateral connections influence neurons
        lateral_injection: Where to inject ('pre_synapse', 'post_nlm', 'both')

    Returns:
        STDPCTM wrapper

    Example:
        >>> from models.ctm import ContinuousThoughtMachine
        >>> from models.ctm_stdp import wrap_ctm_with_stdp
        >>>
        >>> ctm = ContinuousThoughtMachine(...)
        >>> stdp_ctm = wrap_ctm_with_stdp(ctm, stdp_lr=0.0001)
        >>>
        >>> # Use like normal CTM
        >>> predictions, certainties, synch = stdp_ctm(x)
    """
    return STDPCTM(
        base_ctm=ctm,
        stdp_lr=stdp_lr,
        stdp_decay=stdp_decay,
        lateral_strength=lateral_strength,
        lateral_injection=lateral_injection,
    )
