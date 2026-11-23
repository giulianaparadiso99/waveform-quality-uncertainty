# Waveform quality with uncertainty

Classification of strong-motion records of [ESM25](https://esm-db.eu/) as **good** or **bad** quality, with an
estimate of how much each prediction can be trusted.

In the Engineering Strong Motion database every record is reviewed by an analyst before it is processed and
published. This project trains a neural network on the decisions already taken in ESM25, so that new records can be
pre-classified, and attaches to every prediction a measure of uncertainty: confident predictions can be accepted
automatically, uncertain ones sent to manual review.

## Method in brief

- **Input**: spectrogram (STFT) and scalogram (Morlet wavelet transform) of the three components of every record,
  in decibels, on a common grid of 128 logarithmic frequencies and 128 time bins (3 × 128 × 128). The whole stored
  trace (420 s) is used; the zero padding that aligns the traces is ignored by the network through a time mask.
- **Network**: convolutional-recurrent classifier adapted from CRED (residual convolutional stages, bidirectional
  LSTM, mean over time, dense head), trained with a class-weighted binary cross-entropy.
- **Uncertainty**: Monte Carlo dropout (50 stochastic passes): predictive entropy, its aleatoric part and the mutual
  information (epistemic part).
- **Evaluation**: discrimination (AUC, balanced accuracy, average precision), calibration (reliability diagram,
  ECE), error detection by uncertainty and accuracy after rejecting the most uncertain records.
- **Explanations**: Grad-CAM maps over MC dropout passes, compared between categories of records (right or wrong,
  low or high uncertainty, true class).
- **Hyperparameters**: Optuna search (TPE sampler, median pruner) with k-fold cross-validation grouped by event.

## Repository structure

```
.
├── configs/default.yaml   every parameter of the project (data, preprocessing, model, training, evaluation, search)
├── data/                  metadata, development subset, processed features, split (see data/README.md)
├── notebooks/             the pipeline, one notebook per step (see notebooks/README.md)
├── src/wqu/               the Python package used by the notebooks
│   ├── data/              loading of ESM25, waveforms, training/validation split
│   ├── preprocessing/     windowing, transforms, decibel grid, feature files, time masks
│   ├── models/            network, loss, MC dropout, Grad-CAM
│   ├── training/          datasets, training loop, cross-validation experiments, learning curve, search
│   ├── evaluation/        uncertainty assessment, Grad-CAM analysis, padding checks
│   └── visualization.py   plotting functions
├── tests/                 checks of the transforms against SciPy and ObsPy
├── runs/                  trained models, learning curve, hyperparameter search (generated)
└── outputs/               figures and tables of the exploratory notebooks (generated)
```

## Installation

The project was developed with Python 3.11 in a conda environment.

```bash
conda create -n wqu python=3.11
conda activate wqu
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the `wqu` package in editable mode, so that the notebooks import the current code.
Training runs on the CPU, on an NVIDIA GPU or on the Apple GPU (`training.device` in the configuration).

## Usage

1. Set the parameters in `configs/default.yaml`. The keys that select what is run are `data.source`,
   `waveforms.source`, `preprocessing.window.placement`, `model.time_masking`, `training.representation` and
   `training.feature_source`.
2. Run the notebooks in order (see [notebooks/README.md](notebooks/README.md)). The long steps (preprocessing,
   training, search) are resumable: after an interruption, run the notebook again.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests compare the spectrogram with `scipy.signal.spectrogram` and the scalogram with
`obspy.signal.tf_misfit.cwt`, and check the averaging on the logarithmic frequency grid and on the time bins.

## Main references

- ESM25: Cianetti et al. (2026), doi:10.13127/ai/esm25.
- ESM: Luzi et al. (2020), doi:10.13127/esm.2.
- SeisBench: Woollam et al. (2022).
- CRED: Mousavi et al. (2019).
- MC dropout: Gal and Ghahramani (2016).
- Calibration: Guo et al. (2017).
- Grad-CAM: Selvaraju et al. (2017).
- Optuna: Akiba et al. (2019).
