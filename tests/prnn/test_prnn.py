from prnn import *
from trainer import *


settings = {
    'data_path': 'datasets/gpCurves.data',
    'decoder_type': 'SoftLayer',

    'input_norm': False,   # Note: keep false. Normalization has not yet been consistently implemented for computing losses.
    'output_norm': False,

    'mat_points': 2, # What is this?
    'feature_dim': 3,
    'output_dim': 3,
}

from utils import StressStrainDataset

dataset = StressStrainDataset(settings['data_path'], [0,1,2], [3,4,5], seq_length=60)
all_samples = dataset.get_all_batches()
num_samples = len(dataset)
all_indices = np.arange(num_samples)
train_indices = all_indices[:40]
val_indices = all_indices[40:70]
test_indices = all_indices[70:]
val_dataset = all_samples[val_indices]
test_dataset = all_samples[test_indices]

print('test_dataset.shape', test_dataset.shape)


model = PRNN(n_features=settings['feature_dim'], n_outputs=settings['output_dim'], n_matpts=settings['mat_points'], decoder_type=settings['decoder_type'])

print(f'New PRNN: Input (strain) size {model.n_features} - Material layer size (points) {model.n_matpts} - Output (stress) size {model.n_outputs}')

# Create material and yield config for actual use
material = jax_j2.create_material()

# Is this required if we are loading params? I.e. does it modify model
# Initialize model parameters
#params = model.init(jax.random.PRNGKey(0),
#                    jnp.zeros((1, 1, settings['feature_dim'])),  # Dummy input
#                    material,   # Material params
#                    )

def load_params(filename):

    # Add .npy extension if not present
    if not filename.endswith('.npy'):
        filename = f"{filename}.npy"

    # Load the checkpoint
    checkpoint = np.load(filename, allow_pickle=True).item()

    print(f"best_val: {checkpoint['best_val']}")
    print(f"_epoch: {checkpoint['epoch']}")

    if checkpoint['best_params'] is not None:
        # Convert NumPy arrays back to JAX arrays
        best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), checkpoint['best_params'])
        return best_params
    else:
        params = jax.tree_util.tree_map(lambda x: jnp.array(x), checkpoint['params'])
        return params

params = load_params('checkpoints_matpts_2/model_0_ncurve_16.npy')

strain = np.array([[-0.000856862, -0.00292643, -0.00297788], [-0.000856862, -0.00292643, -0.00297788]])
#         [curve index, strain/stress split, sequence, n_features]
data = test_dataset[0:1, 0, 0:2, :]
print(data.shape, data)
# Input of data: (batch, sequence, n_features)
y = model.apply(params, data, material)
print(y)