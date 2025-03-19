

# Outcrop 3D Gaussian Splatting

## Project Overview

OutcropGaussian is an extension of the 3D Gaussian Splatting (3DGS) algorithm, specifically optimized for geological outcrop 3D reconstruction.

## Installation

### Requirements

The basic requirements are consistent with the original 3DGS project:

- Python 3.8+
- PyTorch 1.13+
- CUDA 11.8+

### Installation Steps

1. Clone this repository:
```bash
git clone https://github.com/giseryn/OutcropGaussian.git
cd OutcropGaussian
```

2. Create and activate a virtual environment:
```bash
conda env create --file environment.yml
conda activate OutcropGaussian
```

3. Install dependencies:
```bash
pip install matplotlib
pip install scikit-learn
pip install opencv-python
pip install wandb
pip install huggingface-hub
pip install pandas
```

4. **Important**: Rasterization Code Installation
   
   This project uses a modified version of the rasterization code that needs to be recompiled:
```bash
cd submodules/diff-gaussian-rasterization
pip install -e .
```

5. Install other submodule dependencies:
```bash
cd submodules/simple-knn
pip install -e .
```

## Depth Map Generation

This project uses the Depth Anything model to generate depth maps. To generate depth maps yourself, follow these steps:

1. Install Depth Anything:
```bash
pip install depth-anything
```

2. Use the provided script to generate depth maps:
```bash
python scripts/generate_depth.py --input_dir path/to/images --output_dir path/to/depth_maps
```

## Running Experiments

### Main Experiments

Run the `ablation_experiments.py` script to execute all experiments:

```bash
python ablation_experiments.py --data_dir path/to/dataset
```

### Experiment Details

The script includes the following experiments:

1. **Depth Constraint Comparison**: Compare reconstruction quality with and without depth constraints


2. **Hyperparameter Tuning**: Test the impact of different depth weights on reconstruction quality


3. **Ablation Studies**: Analyze the contribution of each model component



## Acknowledgements

This project is built upon 3D Gaussian Splatting. We thank the original authors for their open-source contributions. We also thank the Depth Anything team for providing an excellent depth estimation model.