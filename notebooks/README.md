# Notebooks

The pipeline runs as a sequence of notebooks. Every parameter is read from `configs/default.yaml`; the notebooks hold no
settings of their own. The code they call is in the `wqu` package (`src/wqu/`).

## Overview

| Notebook | Purpose | Reads | Writes |
|---|---|---|---|
| `01_data_loading_exploration` | Metadata of the two classes, univariate separation of the classes, structure of events and stations, inspection of the waveforms (length, zero padding, gaps, spectra) | ESM25 metadata; traces of `waveforms.source` | `outputs/figures/01_exploration`, `outputs/tables/01_exploration`; the development subset in `data/dev_subset/` on first use |
| `02_preprocessing` | Choice of the analysis window and of the anchoring method on the development subset; spectrogram and scalogram of example records; decibel floor | Development subset | `outputs/*/02_preprocessing`; features of the development subset in `data/processed/dev_subset/` |
| `02b_preprocessing_full` | Spectrograms and scalograms of all ESM25 records, with the window placement of `preprocessing.window.placement` | Full SeisBench datasets | `data/processed/seisbench/[<placement>/]features.h5` and `records.csv` |
| `02c_time_masks` | Time bins of every record that contain recorded data (needed with `model.time_masking: true`) | Feature file, traces | `time_masks.h5` next to the feature file |
| `02d_obspy_features` | Images in the style of Shalabi et al. (ObsPy spectrogram, 64 x 64; settings in `preprocessing.obspy_spectrogram`) and check of the configured network on them | Full SeisBench datasets | `data/processed/seisbench/obspy/<settings>/features.h5`, `records.csv`, `excluded_records.csv` |
| `03_split` | Training/validation split grouped by event and stratified by label | ESM25 metadata | `data/splits/esm25_train_validation.csv`, `outputs/tables/03_split` |
| `04_model_check` | Architecture, number of parameters, training speed, MC dropout and Grad-CAM mechanics on an untrained network | Features of the development subset | `outputs/*/04_model` |
| `05_training` | Training of one model and MC dropout predictions on the validation set | Feature file, time masks, split | `runs/<representation>_<date>_<time>/` |
| `06_evaluation` | Metrics, confusion matrices, reliability diagram, error detection by uncertainty, rejection curves | Validation predictions of a run | `<run>/evaluation/` |
| `07_explanation` | Grad-CAM by prediction category, mean maps and profiles, single records, confident errors, padding checks | Run, feature file, traces | `<run>/explanation/` |
| `08a_fold_choice` | Learning curve and choice of the number of folds k of the search | Feature file, time masks, split | `runs/learning_curve/<variant>/` |
| `08_hyperparameter_search` | Optuna search with k-fold cross-validation; out-of-fold MC dropout predictions of the best trial | Feature file, time masks, split | `runs/search/<study>/` |

`02` and `04` are studies and checks: no later notebook needs their outputs, apart from the window and anchor chosen in
`02`, which are written by hand in the configuration.

## Order

```
01 ──> 02 ──> 02b ──> 02c ──> 03 ──> 05 ──> 06
                                  │       └──> 07
                                  ├──> 08a ──> 08 ──> 05 (best settings, 2-3 seeds)
                                  └──> 04
```

- `02c` is needed only for the network with time masking, and must be run again for every window placement.
- `02d` replaces `02b` and `02c` for the replication of Shalabi et al. (`training.representation: obspy_spectrogram`,
  `model.architecture: shalabi_inception`); it is run again for every combination of image settings.
- `03` is run once: the split is saved and reused by every experiment. The folds of the search (notebook `08`) are
  computed from the same file.
- `06` and `07` evaluate the run chosen with `evaluation.run` (`latest` = most recent run of
  `training.representation`).

## Settings to check before running

| Key | Used by | Meaning |
|---|---|---|
| `training.representation` | 05, 06, 07, 08a, 08 | `spectrogram`, `scalogram` or `obspy_spectrogram` |
| `training.class_weighting`, `training.standardization` | 05 | weighted or unweighted loss; standardized or stored inputs |
| `training.feature_source` | 02c, 05, 07, 08a, 08 | `seisbench` (full dataset) or `dev_subset` |
| `preprocessing.window.placement` | 02, 02b, 02c, 05, 07, 08a, 08 | `anchored`, `clamped` or `full` |
| `model.time_masking` | 02c, 05, 07, 08a, 08 | network that ignores the zero padding |
| `evaluation.run` | 06, 07 | run to evaluate |
| `search.cross_validation_folds` | 08 | number of folds k (1 = the split of `03`) |

The representation, the window placement, the time masking and the number of folds identify an experiment: the
learning curve and the search use them in their folder names (for example
`runs/search/search_scalogram_full_masked_cv3`), so a different value starts a different experiment instead of
resuming the previous one. Notebook `07` checks that the placement and the masking of the configuration are those of
the run.

## Outputs

**Training run** (`runs/<representation>_<date>_<time>/`):

| File | Content |
|---|---|
| `config.yaml` | copy of the configuration of the run |
| `normalization.npz` | per-frequency statistics of the training records |
| `history.csv` | losses and metrics of every epoch |
| `checkpoint.pt` | state at the last epoch (resuming) |
| `best_model.pt` | weights of the epoch with the lowest validation loss |
| `validation_predictions.csv` | deterministic and MC dropout predictions, uncertainty measures |
| `excluded_records.csv` | records without any recorded data (with time masking; classified bad by rule) |
| `evaluation/`, `explanation/` | outputs of notebooks `06` and `07` |

**Hyperparameter search** (`runs/search/<study>/`): `study.db` (Optuna database), one folder `trial_<number>/` per
trial with one subfolder `fold_<k>/` per fold (checkpoint, best model, history, `fold_predictions.csv`),
normalization statistics of every fold, `trials.csv`, `best_settings.json`, `best_trial_predictions.csv` and the
figures of the search.

## Resuming after an interruption

| Notebook | Behaviour |
|---|---|
| `02b` | continues from the records not yet processed |
| `02c` | starts again from the beginning (no incomplete file is left) |
| `02d` | continues from the records not yet processed |
| `05` | with `training.resume: true`, continues the most recent unfinished run from its last epoch |
| `08a` | finished runs are skipped, the interrupted one continues from its last epoch |
| `08` | finished trials and folds are kept; the interrupted trial is resumed from its last epoch after `search.grace_period_s` seconds |

Settings must not change between the sessions of one run or study.

## Approximate durations

On an Apple M-series GPU with the full traces and time masking: one training (`05`) takes about two hours (about
2.5 minutes per epoch); the Grad-CAM step of `07` about ten minutes on the validation set; `08a` five trainings; `08`
with k = 3 and 21 trials several days, depending on how many trials are pruned. Running two notebooks that use the GPU
at the same time slows both.
