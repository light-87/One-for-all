# One-For-All Phosphorylation Site Prediction Pipeline

A comprehensive single-file implementation for phosphorylation site prediction that combines traditional machine learning (XGBoost) with modern transformer architectures and advanced ensemble methods.

## Overview

This pipeline implements state-of-the-art methods for phosphorylation site prediction including:

- **Traditional ML**: XGBoost with engineered features (AAC, DPC, TPC, Binary Encoding, Physicochemical)
- **Deep Learning**: Multiple transformer architectures with novel attention mechanisms
- **Ensemble Methods**: Voting, stacking, dynamic selection, and confidence-weighted ensembles
- **Novel Architectures**: Hierarchical attention and multi-scale fusion transformers
- **Advanced Loss Functions**: Focal loss, motif-aware loss for improved training

## Features

### 🚀 **Performance Optimizations**
- Memory-efficient processing with datatable library
- Mixed precision training for transformers
- Gradient accumulation for larger effective batch sizes
- GPU acceleration for XGBoost
- LRU caching for feature extraction

### 🧠 **Novel Approaches**
- Hierarchical attention mechanism for local and global context
- Multi-scale window fusion for different sequence contexts
- Motif-aware loss function with kinase-specific weighting
- Dynamic ensemble selection based on input similarity
- Confidence-weighted predictions

### 📊 **Comprehensive Analysis**
- Cross-validation with protein-level grouping
- SHAP values for model interpretability
- Attention weight visualization
- Error pattern analysis and clustering
- Feature importance analysis

### 🔧 **Flexibility**
- Modular pipeline with configurable sections
- Multiple model variants to experiment with
- Hyperparameter optimization with Optuna
- Cached intermediate results for faster iteration

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM recommended

### Required Packages
```bash
pip install numpy pandas datatable torch transformers xgboost scikit-learn matplotlib seaborn tqdm pyyaml optuna wandb

# For GPU support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optional for interpretability
pip install shap
```

### Data Requirements
Place the following files in the `data/` directory:
- `Sequence_data.txt` - Protein sequences in FASTA format
- `labels.xlsx` - Phosphorylation site labels
- `physiochemical_property.csv` - Amino acid properties

## Quick Start

### 1. Basic Usage
```bash
# Linux/Mac
./run.sh

# Windows
run.bat
```

### 2. Custom Configuration
```bash
# Use custom config file
./run.sh --config my_config.yaml

# Skip specific steps
./run.sh --skip-preprocessing --skip-features

# Run with cross-validation
./run.sh --cross-validation
```

### 3. Section-by-Section Execution
Edit `config.yaml` to enable/disable specific pipeline sections:
```yaml
pipeline:
  run_data_preprocessing: true
  run_feature_extraction: false  # Skip if features already extracted
  run_xgboost: true
  run_transformers: true
  run_ensemble: true
  run_evaluation: true
  run_cross_validation: false
```

## Configuration

The pipeline is controlled by `config.yaml`. Key sections:

### Data Processing
```yaml
data:
  window_size: 10  # Amino acids on each side of the site
  max_sequence_length: 5000
  balance_classes: true
  random_seed: 42
```

### Feature Extraction
```yaml
features:
  extract_aac: true      # Amino Acid Composition
  extract_dpc: true      # Dipeptide Composition
  extract_tpc: true      # Tripeptide Composition
  extract_binary: true   # Binary encoding
  extract_physicochemical: true
  batch_size: 1000
```

### XGBoost Configuration
```yaml
xgboost:
  enabled: true
  hyperparameter_tuning: false
  params:
    n_estimators: 1000
    max_depth: 6
    learning_rate: 0.1
    device: 'cuda'  # or 'cpu'
```

### Transformer Models
```yaml
transformers:
  enabled: true
  use_mixed_precision: true
  gradient_accumulation_steps: 4
  
  models:
    esm2_small:
      enabled: true
      model_name: "facebook/esm2_t6_8M_UR50D"
      batch_size: 32
      learning_rate: 2e-5
      epochs: 10
    
    hierarchical_attention:
      enabled: true
      use_local_attention: true
      use_global_attention: true
      context_window: 3
```

## Pipeline Sections

### Section 1: Data Preprocessing
- Loads protein sequences and phosphorylation site labels
- Generates balanced negative samples from S/T/Y sites
- Splits data by protein to prevent data leakage
- Cleans and validates data integrity

### Section 2: Feature Extraction
- **AAC**: Amino acid composition in sequence windows
- **DPC**: Dipeptide composition patterns
- **TPC**: Tripeptide composition (memory-optimized)
- **Binary Encoding**: One-hot encoding of amino acids
- **Physicochemical**: Properties like hydrophobicity, charge

### Section 3: XGBoost Training
- Traditional gradient boosting with engineered features
- GPU acceleration and hyperparameter tuning
- Feature importance analysis
- Early stopping and cross-validation

### Section 4: Transformer Training
- **Base ESM-2**: Pre-trained protein language model
- **Hierarchical Attention**: Local motif + global context
- **Multi-Scale Fusion**: Multiple window sizes combined
- Mixed precision training and gradient accumulation

### Section 5: Novel Loss Functions
- **Focal Loss**: Addresses class imbalance
- **Motif-Aware Loss**: Weights based on kinase motifs (SP, TP, etc.)
- **Contrastive Loss**: Learns better representations

### Section 6: Ensemble Methods
- **Voting**: Weighted combination of predictions
- **Stacking**: Meta-learner trained on base predictions
- **Dynamic Selection**: Choose best model per input
- **Confidence Weighting**: Weight by prediction confidence

### Section 7: Evaluation
- Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC, MCC)
- ROC and Precision-Recall curves
- Confusion matrices and error analysis
- Feature importance and attention visualizations

### Section 8: Cross-Validation
- Protein-level stratified group K-fold
- Statistical significance testing
- Per-fold performance analysis

### Section 9: Interpretability
- SHAP values for feature importance
- Attention weight visualization
- Error pattern clustering
- Motif-specific performance analysis

## Output Structure

```
outputs/
├── data/
│   ├── train_data.csv
│   ├── val_data.csv
│   └── test_data.csv
├── features/
│   ├── train_features.csv
│   ├── val_features.csv
│   └── test_features.csv
├── xgboost_predictions.csv
├── xgboost_metrics.json
├── transformer_predictions.csv
├── ensemble_predictions.csv
└── experiment.log

checkpoints/
├── xgboost_model.json
├── best_transformer_esm2.pt
├── hierarchical_attention.pt
└── ensemble_models/
```

## Advanced Usage

### Hyperparameter Tuning
Enable in config:
```yaml
xgboost:
  hyperparameter_tuning: true  # Uses Optuna

transformers:
  hyperparameter_tuning: true  # Grid search over learning rates
```

### Weights & Biases Integration
```yaml
wandb:
  enabled: true
  project: "phospho-prediction"
  entity: "your-username"
  log_model: true
  log_predictions: true
  log_attention_maps: true
```

### Custom Model Architectures
Add new transformer variants in the pipeline:
```python
class CustomTransformer(BasePhosphoTransformer):
    def __init__(self, model_name, custom_params):
        super().__init__(model_name)
        # Custom architecture implementation
        
    def forward(self, input_ids, attention_mask):
        # Custom forward pass
        pass
```

## Performance Tips

### Memory Optimization
- Use datatable for large datasets (5x faster than pandas)
- Enable mixed precision training
- Process features in batches
- Clean up intermediate variables with `gc.collect()`

### Speed Optimization
- Enable GPU acceleration for XGBoost
- Use gradient accumulation for larger effective batch sizes
- Cache feature extraction results
- Use parallel processing for feature extraction

### Model Selection
- Start with XGBoost baseline for quick results
- Use ESM-2 small for faster transformer training
- Enable ensemble methods for best performance
- Use cross-validation for robust evaluation

## Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Reduce batch sizes in config.yaml
transformers:
  models:
    esm2_small:
      batch_size: 16  # Reduce from 32
```

**CUDA Not Available**
```bash
# Install GPU-enabled PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Feature Extraction Too Slow**
```bash
# Enable datatable optimization and increase batch size
features:
  use_datatable: true
  batch_size: 2000  # Increase if memory allows
```

**Model Training Fails**
- Check data file integrity
- Verify sufficient disk space for outputs
- Monitor GPU memory usage
- Check log files for detailed error messages

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{one_for_all_phospho,
  title={One-For-All Phosphorylation Site Prediction Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions and issues:
- Check the troubleshooting section
- Review log files in `outputs/experiment.log`
- Open an issue on GitHub
- Contact the maintainers

## Acknowledgments

- ESM-2 protein language model from Meta AI
- XGBoost library for gradient boosting
- Transformers library from Hugging Face
- Original phosphorylation datasets and benchmarks