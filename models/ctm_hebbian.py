"""
Hebbian CTM: Continuous Thought Machine with Hebbian Lateral Connections

"Neurons that fire together wire together"

This module extends the base CTM with Hebbian learning for lateral connections.
The key insight: CTM already computes synchronization (pairwise neuron products).
We use this as the Hebbian signal to strengthen connections between co-active neurons,
creating "muscle memory" - fast paths that bypass full deliberation for familiar patterns.

Core idea:
- When neurons fire together (high synchronization), strengthen their lateral connection
- These strengthened connections provide a "fast path" for familiar patterns
- Over time, the network develops shortcuts for common activation patterns
"""

import torch
import torch.nn as nn
import math


class HebbianLateralModule(nn.Module):
    """
    Hebbian lateral connection module.

    Maintains a matrix of lateral connection strengths that are updated
    based on neuron co-activation patterns (Hebbian learning).

    Args:
        d_model: Number of neurons
        hebbian_lr: Learning rate for Hebbian updates
        hebbian_decay: Decay factor to prevent unbounded growth (applied each step)
        use_oja: If True, use Oja's rule for normalized Hebbian learning
        lateral_strength: Scaling factor for lateral input contribution
    """
    def __init__(
        self,
        d_model,
        hebbian_lr=0.01,
        hebbian_decay=0.999,
        use_oja=False,
        lateral_strength=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.use_oja = use_oja
        self.lateral_strength = lateral_strength

        # Hebbian weights: accumulated co-activation patterns
        # Using None for lazy initialization to avoid MPS buffer issues
        self._hebbian_weights = None

        # Learnable gate: controls how much lateral input affects each neuron
        # Using simple per-neuron learnable scalars to avoid MPS issues
        # Small random init for diversity; sigmoid(0.01) ≈ 0.5
        self.lateral_gate_weights = nn.Parameter(torch.randn(d_model) * 0.1)

        # Optional: learnable base lateral weights (sparse initialization)
        # These provide a learned "prior" on connectivity
        self.register_parameter(
            'base_lateral',
            nn.Parameter(torch.zeros(d_model, d_model))
        )

        # Pre-compute diagonal mask (1s everywhere except diagonal)
        # This avoids creating torch.eye() during forward pass which can cause MPS issues
        self.register_buffer('_diag_mask', 1.0 - torch.eye(d_model))

    @property
    def hebbian_weights(self):
        """Lazy initialization of hebbian weights on first access."""
        if self._hebbian_weights is None:
            # Initialize on same device as base_lateral
            self._hebbian_weights = torch.zeros(
                self.d_model, self.d_model,
                device=self.base_lateral.device,
                dtype=self.base_lateral.dtype
            )
        return self._hebbian_weights

    @hebbian_weights.setter
    def hebbian_weights(self, value):
        self._hebbian_weights = value

    def get_effective_weights(self):
        """
        Combine base (learned) weights with Hebbian (accumulated) weights.
        """
        # Normalize Hebbian weights to prevent explosion
        hebb_normalized = self.hebbian_weights
        max_val = self.hebbian_weights.abs().max()
        if max_val > 0:
            hebb_normalized = self.hebbian_weights / (max_val + 1e-8)

        effective = self.base_lateral + hebb_normalized

        # Zero diagonal (no self-connections) - use pre-computed mask
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
        lateral_input = torch.matmul(activated_state, effective_weights.T)  # (B, D)

        # Apply per-neuron learnable gate
        gate = torch.sigmoid(self.lateral_gate_weights)  # (D,)
        gated_lateral = gate * lateral_input * self.lateral_strength
        return gated_lateral

    def hebbian_update(self, activated_state):
        """
        Update Hebbian weights based on co-activation.

        Standard Hebbian: ΔW_ij = η * act_i * act_j
        Oja's rule: ΔW_ij = η * act_i * (act_j - W_ij * act_i)

        Args:
            activated_state: (B, D) current neuron activations
        """
        if not self.training:
            return

        with torch.no_grad():
            # Mean activation across batch for stability
            act = activated_state.mean(dim=0)  # (D,)

            if self.use_oja:
                # Oja's rule: normalized Hebbian that prevents explosion
                # ΔW_ij = η * act_i * (act_j - W_ij * act_i)
                act_col = act.unsqueeze(1)  # (D, 1)
                act_row = act.unsqueeze(0)  # (1, D)

                reconstruction = self.hebbian_weights * act_col  # (D, D)
                error = act_row - reconstruction
                delta = self.hebbian_lr * act_col * error
            else:
                # Standard Hebbian: outer product of activations (MPS-friendly)
                delta = self.hebbian_lr * (act.unsqueeze(1) @ act.unsqueeze(0))

            # Decay old weights and add new (non-in-place for MPS compatibility)
            new_weights = self.hebbian_weights * self.hebbian_decay + delta

            # Zero diagonal - use pre-computed mask
            self._hebbian_weights = new_weights * self._diag_mask

    def reset_hebbian(self):
        """Reset Hebbian weights to zero (useful for new episodes)."""
        if self._hebbian_weights is not None:
            self._hebbian_weights = torch.zeros_like(self._hebbian_weights)

    def get_hebbian_stats(self):
        """Return statistics about Hebbian weights for monitoring."""
        with torch.no_grad():
            weights = self.hebbian_weights
            return {
                'mean': weights.mean().item(),
                'std': weights.std().item(),
                'max': weights.max().item(),
                'min': weights.min().item(),
                'sparsity': (weights.abs() < 1e-6).float().mean().item(),
                'top_10_mean': weights.abs().flatten().topk(10).values.mean().item(),
            }


class HebbianCTM(nn.Module):
    """
    CTM with Hebbian Lateral Connections.

    Wraps an existing CTM and adds:
    1. Hebbian lateral connection module
    2. Integration of lateral input into the forward pass
    3. Hebbian weight updates based on synchronization

    The lateral connections provide a "fast path" for familiar patterns:
    - Neurons that frequently co-activate develop strong lateral connections
    - These connections allow direct influence without going through full synapse computation
    - Results in faster convergence for familiar/repeated patterns ("muscle memory")

    Args:
        base_ctm: An existing ContinuousThoughtMachine instance
        hebbian_lr: Learning rate for Hebbian updates
        hebbian_decay: Decay factor for Hebbian weights
        use_oja: Use Oja's rule instead of standard Hebbian
        lateral_strength: How strongly lateral input affects neurons
        lateral_injection: Where to inject lateral input ('pre_synapse', 'post_nlm', 'both')
    """
    def __init__(
        self,
        base_ctm,
        hebbian_lr=0.01,
        hebbian_decay=0.999,
        use_oja=False,
        lateral_strength=0.1,
        lateral_injection='pre_synapse',
    ):
        super().__init__()

        self.ctm = base_ctm
        self.lateral_injection = lateral_injection

        # Create Hebbian lateral module
        self.hebbian = HebbianLateralModule(
            d_model=base_ctm.d_model,
            hebbian_lr=hebbian_lr,
            hebbian_decay=hebbian_decay,
            use_oja=use_oja,
            lateral_strength=lateral_strength,
        )

        # If injecting pre-synapse, we need to adjust synapse input size
        # The synapse expects [attn_out, activated_state]
        # We'll add lateral_input to activated_state before concatenation

    def forward(self, x, track=False):
        """
        Forward pass with Hebbian lateral connections.

        Same interface as base CTM, but with lateral connections active.
        """
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []
        hebbian_stats_tracking = []

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

            # === HEBBIAN LATERAL INPUT (pre-synapse injection) ===
            if self.lateral_injection in ('pre_synapse', 'both'):
                lateral_input = self.hebbian.compute_lateral_input(activated_state)
                # Add lateral input to activated state before synapse
                activated_state_for_synapse = activated_state + lateral_input
            else:
                activated_state_for_synapse = activated_state

            pre_synapse_input = torch.cat((attn_out, activated_state_for_synapse), dim=-1)

            # --- Apply Synapses ---
            state = self.ctm.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply Neuron-Level Models ---
            activated_state = self.ctm.trace_processor(state_trace)

            # === HEBBIAN LATERAL INPUT (post-NLM injection) ===
            if self.lateral_injection in ('post_nlm', 'both'):
                lateral_input = self.hebbian.compute_lateral_input(activated_state)
                activated_state = activated_state + lateral_input

            # === HEBBIAN UPDATE ===
            # Update lateral weights based on current co-activations
            self.hebbian.hebbian_update(activated_state)

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
                hebbian_stats_tracking.append(self.hebbian.get_hebbian_stats())

        # --- Return Values ---
        if track:
            import numpy as np
            # Return same format as base CTM for compatibility
            # hebbian_stats can be accessed via model.hebbian.get_hebbian_stats()
            return (
                predictions,
                certainties,
                (np.array(synch_out_tracking), np.array(synch_action_tracking)),
                np.array(pre_activations_tracking),
                np.array(post_activations_tracking),
                np.array(attention_tracking),
            )
        return predictions, certainties, synchronisation_out

    def reset_hebbian(self):
        """Reset Hebbian weights (call between episodes if desired)."""
        self.hebbian.reset_hebbian()

    def get_hebbian_weights(self):
        """Get current Hebbian weight matrix for visualization."""
        return self.hebbian.hebbian_weights.detach().cpu()

    def get_effective_lateral_weights(self):
        """Get combined base + Hebbian weights."""
        return self.hebbian.get_effective_weights().detach().cpu()


def wrap_ctm_with_hebbian(
    ctm,
    hebbian_lr=0.01,
    hebbian_decay=0.999,
    use_oja=False,
    lateral_strength=0.1,
    lateral_injection='pre_synapse',
):
    """
    Convenience function to wrap an existing CTM with Hebbian lateral connections.

    Args:
        ctm: Existing ContinuousThoughtMachine instance
        hebbian_lr: Learning rate for Hebbian updates (higher = faster learning)
        hebbian_decay: Decay rate (closer to 1 = longer memory)
        use_oja: Use Oja's normalized rule (more stable)
        lateral_strength: How much lateral connections influence neurons
        lateral_injection: Where to inject ('pre_synapse', 'post_nlm', 'both')

    Returns:
        HebbianCTM wrapper

    Example:
        >>> from models.ctm import ContinuousThoughtMachine
        >>> from models.ctm_hebbian import wrap_ctm_with_hebbian
        >>>
        >>> ctm = ContinuousThoughtMachine(...)
        >>> hebbian_ctm = wrap_ctm_with_hebbian(ctm, hebbian_lr=0.01)
        >>>
        >>> # Use like normal CTM
        >>> predictions, certainties, synch = hebbian_ctm(x)
    """
    return HebbianCTM(
        base_ctm=ctm,
        hebbian_lr=hebbian_lr,
        hebbian_decay=hebbian_decay,
        use_oja=use_oja,
        lateral_strength=lateral_strength,
        lateral_injection=lateral_injection,
    )
