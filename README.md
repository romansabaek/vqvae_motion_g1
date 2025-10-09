# MotionVQVAE

A PyTorch implementation of Motion Vector Quantized Variational Autoencoder for motion data compression and generation.

## Project Structure

```
MotionVQVAE/
├── motion_vqvae/           # Main package
│   ├── __init__.py
│   ├── agent.py           # Main training agent
│   ├── config_loader.py   # YAML configuration loader
│   ├── models/            # Model components
│   │   ├── __init__.py
│   │   ├── models.py      # Main VQVAE models
│   │   ├── encdec.py      # Encoder/Decoder
│   │   ├── quantize_cnn.py # Quantization modules
│   │   └── resnet.py      # ResNet blocks
│   ├── data/              # Data handling
│   │   ├── __init__.py
│   │   ├── vq_dataset.py  # Dataset class
│   │   └── motion_data_adapter.py # Motion data loading
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── utils.py       # Loss functions
│       ├── torch_utils.py # PyTorch utilities
│       ├── motion_lib.py  # PKL motion library
│       ├── motion_lib_npy.py # NPY motion library
│       └── indexing.py    # Joint indexing utilities
├── configs/               # Configuration files
│   └── agent.yaml         # Main configuration file
├── scripts/               # Training scripts
│   └── train_vqvae.py     # Direct training from motion files
└── README.md
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install torch torchvision numpy wandb pyyaml tqdm
```

## Usage

### Training

```bash
# Train directly from motion file
python scripts/train_vqvae.py --motion_file /path/to/your/motion.npy
```

### Basic Training (Legacy)

```python
from motion_vqvae.agent import MVQVAEAgent
from motion_vqvae.config_loader import ConfigLoader

# Load configuration from YAML
config_loader = ConfigLoader('configs/agent.yaml')
config = config_loader.to_dict()

# Create agent
agent = MVQVAEAgent(config=config)

# Setup with motion file
agent.setup_from_file('/path/to/motion.npy')

# Start training
agent.fit()
```

## Configuration

### YAML Configuration

The system supports YAML configuration files with the following structure:

```yaml
agent:
  config:
    # Training parameters
    batch_size: 256
    window_size: 32
    total_iter: 10000
    lr: 2e-4
    
    # Model parameters
    code_dim: 512
    nb_code: 512
    num_joints: 24
    
    # Data processing
    prefer_expmap: false
    apply_sim_to_real_map: false
    real_dof: null
    sim_to_real_idx: null
```

### Data Processing Options

- `prefer_expmap`: Convert quaternions to exponential maps
- `apply_sim_to_real_map`: Apply joint mapping from simulation to real robot
- `real_dof`: Number of real robot degrees of freedom
- `sim_to_real_idx`: Mapping from simulation joints to real joints

## Supported Data Formats

### NPY Files
- Continuous trajectory data with transition flags
- Format: `[time, root_pos(3), root_rot(4), dof_pos(N), transition_flag(1)]`
- Automatically segmented into motion clips

### PKL Files
- Single or multi-motion pickle files
- Standard motion capture format
- Support for SMPL joint data

## Model Architecture

- **Encoder**: 1D CNN with ResNet blocks
- **Quantizer**: Vector quantization with EMA reset
- **Decoder**: Upsampling CNN with ResNet blocks
- **Loss**: Reconstruction + Velocity + Commitment losses

## Data Preparation

The VQ-VAE training uses self-supervised learning where the input and target are the same motion data. Here's how data pairs are created:

1. **Raw Motion Loading**: Load PKL/NPY motion files
2. **Feature Extraction**: Extract root position, rotation, velocities, and joint data
3. **Window Creation**: Create fixed-size windows (e.g., 32 frames) from motion sequences
4. **Normalization**: Apply Z-score normalization
5. **Training Pairs**: Input = Target (same motion window)

For detailed information, see [Data Preparation Guide](docs/data_preparation.md).

## Data Preparation

For detailed data preparation commands, see [Data Preparation Guide](docs/data_preparation.md).

## Features

- **Flexible Data Loading**: Support for PKL and NPY motion files
- **YAML Configuration**: Easy parameter management
- **Joint Mapping**: Support for simulation-to-real robot mapping
- **Multiple Quantizers**: EMA, Reset, EMA-Reset options
- **Wandb Integration**: Optional experiment tracking
- **Checkpoint Management**: Save/load training states
- **Motion Reconstruction**: Generate motion from codebook sequences
