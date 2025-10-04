import jax
import jax.numpy as jnp
import flax.linen as nn
from .jax_j2 import *


class SoftLayer(nn.Module):
    """Custom decoder layer that applies softplus to weights before the linear transform."""
    n_matpts: int   # m: Number of material points
    n_outputs: int  # o: Number of output components
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        # x shape: [b, s, m * o] = [b, s, p]
        # b = batch size
        # s = sequence length
        # p = layer input: m * o

        # Create unconstrained weights parameter
        raw_weights = self.param('raw_weights',
                                 nn.initializers.he_uniform(dtype=jnp.float32),
                                 (self.n_matpts * self.n_outputs,self.n_outputs))   # [p, o]

        # Apply softplus to ensure weights are positive
        weights = nn.softplus(raw_weights)  # [p, o]

        # linear transform, essentially dot product between weights and input
        # Compute using Einstein summation convention
        output = jnp.einsum('bsp,po->bso', x, weights)  # [b, s, o]

        if self.use_bias:
            bias = self.param('bias',
                              nn.initializers.zeros,
                              (self.n_outputs,))
            output = output + bias

        return output


class SparseNormLayer(nn.Module):
    """Custom decoder sparse layer with constrained connectivity pattern and weights.

    Each material point's output is connected with a weight to the corresponding PRNN output comonent.
    All weights are positive, and per component sum to one.
    """
    n_matpts: int   # m: Number of material points
    n_outputs: int  # o: Number of output components

    @nn.compact
    def __call__(self, x):
        # x shape: [b, s, m * o]
        b = x.shape[0]  # batch size
        s = x.shape[1]  # sequence length

        # Create weights parameter
        raw_weights = self.param('raw_weights',
                                 nn.initializers.he_uniform(dtype=jnp.float32),
                                 (self.n_matpts,self.n_outputs))

        # Apply softplus to ensure weights are positive
        weights = nn.softplus(raw_weights)  # [m, o]
        # Scale weights to sum to one component wise
        weights = weights / jnp.sum(weights, axis=0)  # [m, o]

        # # Alternative using softmax
        # weights = jax.nn.softmax(raw_weights, axis=0)   # [m, o]

        # Reshape input for sparse multiplication
        x_reshaped = x.reshape(b, s, self.n_matpts, self.n_outputs)  # [b, s, m, o]

        # Compute layer using Einstein summation convention
        # bsmo: input dimensions
        # mo: weight dimensions (broadcasts to match input)
        # -> sbo: output dimensions (sum over 'm')
        output = jnp.einsum('bsmo,mo->bso', x_reshaped, weights)

        return output


class SparseSharedNormLayer(nn.Module):
    """Custom decoder sparse layer with constrained connectivity pattern and weights.

    Each material point's 3 outputs are connected to the 3 PRNN outputs with a single weight.
    All weights are positive and sum to one.
    """
    n_matpts: int   # m: Number of material points
    n_outputs: int  # o: Number of output components

    @nn.compact
    def __call__(self, x):
        # x shape: [b, s, m * o]
        b = x.shape[0]  # batch size
        s = x.shape[1]  # sequence length

        # Create unconstrained weights parameter
        raw_weights = self.param('raw_weights',
                                 nn.initializers.uniform(dtype=jnp.float32),    # NOTE: uniform distribution
                                 (self.n_matpts,))  # [m]

        # Apply softplus to ensure weights are positive
        weights = nn.softplus(raw_weights)  # [m]
        # Scale weights to sum to one
        weights = weights / jnp.sum(weights)  # [m]

        # Alternative using softmax
        # weights = nn.softmax(raw_weights)  # [m]

        # Reshape input for sparse multiplication
        x_reshaped = x.reshape(b, s, self.n_matpts, self.n_outputs)  # [b, s, m, o]

        # Compute layer using Einstein summation convention
        # sbmo: input dimensions
        # m: weight dimensions (broadcasts to match input)
        # -> bso: output dimensions (sum over 'p')
        output = jnp.einsum('bsmo,m->bso', x_reshaped, weights)  # [b, s, o]

        return output


class PRNN(nn.Module):
    """Physics-regularized neural network using JAX."""
    n_features: int     # f: number of input features (strain tensor components)
    n_outputs: int      # o: Number of output components
    n_matpts: int       # m: Number of material points
    decoder_type: str   # Type of decoder layer to use

    def setup(self):
        # Calculate total size of material layer
        self.n_latents = self.n_matpts * self.n_features

        # First linear layer (without bias) - encoder/localization
        self.encoder = nn.Dense(features=self.n_latents, use_bias=False, name="Encoder")

        # Decoder / homogenization layer
        if self.decoder_type == 'SoftLayer':
            # SoftLayer (Standard layer with softplus on weights)
            self.decoder = SoftLayer(n_matpts=self.n_matpts, n_outputs=self.n_outputs, name="Decoder", use_bias=False)
        elif self.decoder_type == 'SparseNormLayer':
            # Sparse, non-shared weights. Component-wise weights sum to one
            self.decoder = SparseNormLayer(n_matpts=self.n_matpts, n_outputs=self.n_outputs, name="Decoder")
        elif self.decoder_type == 'SparseSharedNormLayer':
            # Sparse, shared weights per material point, positive, and all weights sum to one
            self.decoder = SparseSharedNormLayer(n_matpts=self.n_matpts, n_outputs=self.n_outputs, name="Decoder")
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")


    def __call__(self, x_bf, hist_state, material):
        # x: [b, f]
        b = x_bf.shape[0]  # batch size
        f = x_bf.shape[1]
        #init_hist_state = init_history(b * self.n_matpts)

        # Encoder network:
        # Input: x [b, s, f]
        # Output: strains [b, s, n_latents]
        strains_bl = self.encoder(x_bf.reshape(b, 1, f)).reshape(b, self.n_latents)

        strain_batch = jnp.reshape(strains_bl, (b * self.n_matpts, self.n_features))
        print('hist_state', hist_state.shape)
        # Material model update
        n_isvs = hist_state.shape[-1] // self.n_matpts
        stress_batch, new_hist_state = constitutive_update_batch(
            strain_batch, hist_state.reshape(b * self.n_matpts, n_isvs), material)

        stress_bl = jnp.reshape(stress_batch, (b, self.n_latents))

        # Decoder network:
        # Input: stresses_seq [b, s, expected_latents_from_stress]
        # Output: outputs [b, s, n_outputs]
        stress = self.decoder(stress_bl.reshape(b, 1, self.n_latents)).reshape(b, self.n_outputs)

        return stress, new_hist_state


def create_prnn_model(n_features=3, n_outputs=3, n_matpts=8, random_key=jax.random.PRNGKey(0), decoder_type='SoftLayer'):
    """Create and initialize a PRNN model with material parameters."""
    # Create model
    model = PRNN(n_features=n_features, n_outputs=n_outputs, n_matpts=n_matpts, decoder_type=decoder_type)

    print(f'New PRNN: Input (strain) size {model.n_features} - Material layer size (points) {model.n_matpts} - Output (stress) size {model.n_outputs}')

    # Create material and yield config for actual use
    material = create_material()

    # Initialize model parameters
    params = model.init(random_key,
                       jnp.zeros((1, 1, n_features)),  # Dummy input
                       material,   # Material params
                       )

    return model, params, material