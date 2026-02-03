# Hybrid Quantum-Classical Classifier for Flood Detection

A hybrid quantum-classical deep learning model for semantic segmentation of flood-affected satellite imagery using the FloodNet dataset. This project combines classical Convolutional Neural Networks (CNNs) with quantum computing circuits to classify satellite images into 10 different land-use categories, enabling automated flood damage assessment.

## 🌟 Features

- **Hybrid Architecture**: Combines classical CNNs (feature extraction) with quantum circuits (feature processing)
- **Semantic Segmentation**: Pixel-level classification for flood detection
- **Quantum Computing**: Leverages PennyLane for quantum circuit implementation
- **Multiple Quantum Encodings**: Support for angle, amplitude, IQP, and basis encoding
- **Various Circuit Types**: Strongly entangling, basic entangling, hardware-efficient, and custom circuits
- **PyTorch Lightning**: Modern training framework with automatic checkpointing and logging
- **Hyperparameter Tuning**: Integrated Optuna for automated hyperparameter optimization
- **Visualization Tools**: Comprehensive visualization of results, predictions, and training metrics
- **Test Suite**: Pytest-based tests for data pipeline validation

## 📋 Table of Contents

- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🗂️ Dataset

This project uses the **FloodNet Dataset v1.0**, a high-quality labeled dataset of aerial imagery captured by drones during Hurricane Harvey in Houston, Texas.

### Dataset Classes (10 categories):

0. **Background**: Areas without specific land use
1. **Building-Flooded**: Flooded buildings
2. **Building-Non-Flooded**: Non-flooded buildings
3. **Road-Flooded**: Flooded roads
4. **Road-Non-Flooded**: Non-flooded roads
5. **Water**: Water bodies (rivers, lakes, pools)
6. **Tree**: Trees and vegetation
7. **Vehicle**: Cars, trucks, and other vehicles
8. **Pool**: Swimming pools
9. **Grass**: Grass and lawns

### Dataset Structure:
- **Train Set**: RGB images with corresponding segmentation masks
- **Validation Set**: For hyperparameter tuning
- **Test Set**: For final model evaluation
- **Image Size**: Variable (resized to 640x640 by default)
- **Format**: JPG for images, PNG for masks

The dataset will be automatically downloaded when you run the project for the first time.

## 🏗️ Architecture

### Overview

TODO: Add architecture diagram here

### Components

1. **CNN Feature Extractor**: 
   - Pre-trained ResNet backbone
   - Extracts high-level spatial features from satellite images
   - Output: 256-dimensional feature vector per image

2. **Projection Layer**:
   - Dense layer with ReLU activation
   - Maps CNN features to quantum circuit input dimension
   - Includes batch normalization and dropout for regularization

3. **Quantum Circuit**:
   - Number of qubits: 8 (configurable)
   - Number of layers: 3 (configurable)
   - Encoding: Angle encoding (configurable: angle, amplitude, IQP, basis)
   - Circuit type: Strongly entangling (configurable)
   - Outputs: 8 expectation values in range [-1, 1]

4. **Classical Classifier**:
   - Fully connected layers
   - Maps quantum outputs to class probabilities
   - Uses softmax activation for multi-class classification

### Quantum Encoding Strategies

- **Angle Encoding**: Each feature becomes a rotation angle on a qubit
- **Amplitude Encoding**: Features encode quantum state amplitudes (requires 2^n features)
- **IQP Encoding**: Instantaneous Quantum Polynomial encoding for good expressiveness
- **Basis Encoding**: Binary features encode basis states

### Circuit Types

- **Strongly Entangling**: All-to-all qubit connectivity (most expressive)
- **Basic Entangling**: Nearest-neighbor connectivity (hardware-friendly)
- **Hardware Efficient**: Optimized for real quantum hardware
- **Custom Alternating**: Custom design with alternating rotation and entanglement layers

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher (preferred 3.11)
- CUDA-compatible GPU (recommended, but not required)
- Conda or virtualenv (recommended)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/iLavaU/proyecto-grupo-a-qc.git
cd proyecto-grupo-a-qc
```

2. **Create a virtual environment**:
```bash
# Using conda
conda create -n quantum-flood python=3.11
conda activate quantum-flood

# Or using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Dependencies

Core packages:
- `torch>=2.0.0` - PyTorch for deep learning
- `torchvision>=0.15.0` - Computer vision utilities
- `pytorch-lightning>=2.0.0` - Training framework
- `pennylane>=0.35.0` - Quantum computing framework
- `pennylane-lightning>=0.35.0` - Fast quantum simulator
- `optuna>=3.0.0` - Hyperparameter optimization
- `matplotlib`, `seaborn` - Visualization
- `pytest` - Testing framework

See [requirements.txt](requirements.txt) for the complete list.

## 💻 Usage

### Basic Training

Run the main training script:

```bash
python main.py
```

This will:
1. Download the FloodNet dataset (if not already present)
2. Load and preprocess the data
3. Create train/validation/test splits
4. Initialize the quantum device and hybrid model
5. Train the model with early stopping
6. Evaluate on the test set
7. Generate visualizations and save results

### Hyperparameter Tuning

To find optimal hyperparameters using Optuna:

```bash
python tune_hyperparameters.py
```

This will run a hyperparameter search optimizing:
- Learning rate
- Batch size
- Number of qubits
- Number of quantum layers
- CNN output dimension
- Dropout rate
- Encoding type
- Circuit type

Results will be saved to `hyperparameter_tuning/results/`.

### Custom Configuration

Modify `config.py` to customize the experiment:

```python
from config import Config

# Create custom configuration
config = Config()
config.N_QUBITS = 12          # Use 12 qubits
config.N_LAYERS = 5           # 5 quantum layers
config.BATCH_SIZE = 64        # Larger batch size
config.LEARNING_RATE = 0.0001 # Lower learning rate
config.MAX_EPOCHS = 100       # More epochs
```

## ⚙️ Configuration

All hyperparameters are centralized in `config.py`:

### Quantum Configuration
```python
DEVICE = 'default.qubit'              # Quantum device
N_QUBITS = 8                          # Number of qubits
N_LAYERS = 3                          # Variational layers
ENCODING = 'angle'                    # Encoding strategy
CIRCUIT_TYPE = 'strongly_entangling'  # Circuit architecture
```

### Training Configuration
```python
BATCH_SIZE = 32                       # Training batch size
LEARNING_RATE = 0.001                 # Adam learning rate
MAX_EPOCHS = 50                       # Maximum epochs
PATIENCE = 10                         # Early stopping patience
```

### Data Configuration
```python
MAX_IMAGES_PER_CLASS = 300            # Max images per class
IMAGE_SIZE = (640, 640)               # Image dimensions
TRAIN_RATIO = 0.7                     # 70% training
VAL_RATIO = 0.15                      # 15% validation
TEST_RATIO = 0.15                     # 15% testing
NUM_WORKERS = 2                       # Data loader workers
```

### Model Configuration
```python
CNN_OUTPUT_DIM = 256                  # CNN feature dimension
DROPOUT_RATE = 0.3                    # Dropout probability
```

For detailed configuration options, see [config.py](config.py).

## 📁 Project Structure

```
proyecto-grupo-a-qc/
│
├── data/                          # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataset.py                 # PyTorch Dataset class
│   └── loader.py                  # DataLoader creation
│
├── models/                        # Model architectures
│   ├── __init__.py
│   ├── cnn_extractor.py           # CNN feature extractor
│   └── hybrid_model.py            # Hybrid quantum-classical model
│
├── quantum/                       # Quantum computing components
│   ├── __init__.py
│   ├── circuits.py                # Quantum circuit definitions
│   ├── device.py                  # Quantum device setup
│   └── encoding.py                # Data encoding strategies
│
├── training/                      # Training utilities
│   ├── __init__.py
│   └── trainer.py                 # Training, testing, and evaluation
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   └── visualization.py           # Plotting and visualization
│
├── hyperparameter_tuning/         # Hyperparameter optimization
│   ├── __init__.py
│   └── optuna_tuner.py            # Optuna-based tuning
│
├── tests/                         # Test suite
│   └── test_dataloader.py         # DataLoader tests
│
├── documentation/                 # Project documentation
│   ├── Link carpeta compartida.txt
│   ├── Link dataset.txt
│   └── Link reporte.txt
│
├── FloodNet/                      # Dataset directory (auto-downloaded)
│
├── config.py                      # Configuration file
├── main.py                        # Main training script
├── tune_hyperparameters.py        # Hyperparameter tuning script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## 🧪 Testing

Run the test suite to validate the data pipeline:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_dataloader.py -v

# Run specific test
pytest tests/test_dataloader.py::TestFloodNetDataset::test_label_transform -v

# Run with coverage
pytest tests/ --cov=data --cov-report=html
```

### Test Coverage

The test suite includes:
- ✅ Dataset initialization and loading
- ✅ Image and mask transformations
- ✅ DataLoader batch iteration
- ✅ Train/validation/test splits
- ✅ Data type and shape validation
- ✅ Transform pipeline correctness
- ✅ No data leakage between splits

## 📊 Results

### Performance Metrics

TODO: Add performance metrics table here

### Output Files

Training generates the following outputs:
- `checkpoints/`: Model checkpoints (best and last)
- `logs/`: TensorBoard logs for training visualization
- `results/`: Visualization plots and metrics
- `hyperparameter_tuning/results/`: Optuna study results

### Visualization

The project includes comprehensive visualization tools:
- Sample image and mask visualization
- Training/validation loss curves
- Confusion matrices
- Per-class accuracy metrics
- Prediction examples with ground truth comparison
- Quantum circuit diagrams

## 🔬 How It Works

### Data Flow

TODO: Add data flow diagram here...

1. **Image Loading**: RGB satellite images are loaded from disk
2. **Preprocessing**: 
   - Images resized to 640x640
   - Normalized using ImageNet statistics
   - Converted to PyTorch tensors (float32)
3. **Mask Loading**:
   - Segmentation masks loaded as grayscale images
   - Resized using NEAREST interpolation (preserves class labels)
   - Converted to integer tensors (int64)
4. **Batching**: DataLoader creates batches for efficient training

### Training Process

TODO: Add training flow diagram here...

1. **Forward Pass**:
   - Image → CNN → Feature vector (256-dim)
   - Feature vector → Projection → Quantum input (8-dim)
   - Quantum input → Quantum circuit → Quantum output (8-dim)
   - Quantum output → Classifier → Class probabilities (10-dim)

2. **Loss Calculation**:
   - Cross-entropy loss for semantic segmentation
   - Backpropagation through quantum circuit (using parameter-shift rule)

3. **Optimization**:
   - Adam optimizer with learning rate scheduling
   - Early stopping based on validation loss

4. **Evaluation**:
   - Accuracy, precision, recall, F1-score
   - Per-class metrics
   - Confusion matrix

### Quantum Advantage

The quantum circuit provides:
- **Non-linear transformations**: Through quantum gates and entanglement
- **High-dimensional feature space**: Exponentially large Hilbert space
- **Inherent regularization**: Quantum noise acts as implicit regularization
- **Novel feature representations**: Different from classical neural networks

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Write tests for new features
- Update documentation as needed

## 📄 License

This project is part of academic research at Universidad de la República, Uruguay.

## 📚 References

### Dataset
- **FloodNet Dataset**: [FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021)

### Quantum Computing
- **PennyLane**: [PennyLane Documentation](https://pennylane.ai/)
- **Quantum Machine Learning**: Schuld, M., & Petruccione, F. (2018). *Supervised Learning with Quantum Computers*

### Deep Learning
- **PyTorch**: [PyTorch Documentation](https://pytorch.org/docs/)
- **PyTorch Lightning**: [Lightning Documentation](https://lightning.ai/docs/)

## 👥 Authors

**Grupo A**
TODO: Add author names and affiliations here...


## 🙏 Acknowledgments

- FloodNet dataset creators
- PennyLane quantum computing team
- PyTorch and PyTorch Lightning communities
- Universidad de la República for computational resources

---

**Note**: This is a research project exploring the application of hybrid quantum-classical machine learning to satellite imagery analysis. Results should be validated before production use.

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub or contact the repository maintainers.

---

## 💻 Quick commands

- To train the model:
  ```bash
  python main.py
  ```


*Last updated: February 2026*
