# Unified Track and Version Identification

Official implementation of **"Unified Music Identification for Tracks and Versions"** (ISMIR 2026).

A single embedding model handles both *track identification* (finding the exact recording a short, possibly degraded query came from) and *version identification* (finding other renditions of the same underlying musical work).
Audio is embedded segment-by-segment, and the same embedding space serves both retrieval tasks.

> **Note:** retrieval is GPU-only. The similarity search is built on [cuVS](https://github.com/rapidsai/cuvs) and there is no CPU fallback.

## Overview

| | |
|---|---|
| Input | 16 kHz mono audio, 20 s context windows |
| Front-end | CQT — 12 bins/octave, 8 octaves extracted (7 used), `fmin` 32.7 Hz, 20 ms hop |
| Encoder | CLEWS-style IBN-ResNet with learnable GeM pooling |
| Embedding | 1024-d, L2-normalized |
| Training | Triplet loss with hard positive/negative mining, Adam + cosine annealing |
| Retrieval | cuVS IVF-Flat, or exhaustive search for smaller databases |

Training targets are built from *segment-level cliques*: rather than treating whole tracks as positives, the pipeline locates the regions of two versions that actually correspond, and trains on those.

## Installation

Python 3.11 (developed on 3.11.13).

```bash
./install.sh
conda activate unified
```

This creates a conda environment and installs the project into it, with the dependencies declared in `pyproject.toml`. Conda is used only for `ffmpeg`, which torchaudio needs at the system level. The install is editable, so your edits take effect without reinstalling.

Run the commands below from the repository root.

`environment.lock.yml` is a full `conda env export` of our environment. It is a lock file rather than an install path — the build strings pin it to linux-64 with CUDA 12.8 — and it is provided so the exact environment can be reproduced if needed.

## Model weights

The pre-trained checkpoint is on Zenodo:

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22016080-blue)](https://doi.org/10.5281/zenodo.22016080)

<https://doi.org/10.5281/zenodo.22016080>

| File | Description |
|---|---|
| `epoch-8-step-90000.ckpt` | The checkpoint. Pass its path to `inference.py` as the `ckpt` argument |
| `hparams.yaml` | The configuration the model was trained with |
| `metrics.csv` | Training and validation curves for the run |

Selected at step 90000 as the maximum of the composite validation metric. It is
a full Lightning checkpoint, so it can also be resumed from with
`train.py --ckpt`.

Note that the dataset paths recorded inside the checkpoint are the ones from our
cluster. They are inert for `inference.py` and for resuming training, which
reads its configuration from the config file you pass on the command line, but
`validate.py` builds its dataloaders from the embedded configuration and will
need them pointed at your own copies of the data.

## Quickstart: extracting embeddings

Do not load the bare PyTorch model. The Lightning module's `extract_embeddings()` handles segmentation, padding and pooling the way the model expects, and `inference.py` drives it correctly — including multi-GPU.

```bash
python inference.py <audio_dir_or_file> <checkpoint.ckpt> <output_dir>
```

The architecture and audio front-end are always read from the checkpoint, so they cannot drift from the weights. Only the parameters that are genuinely free at inference time are exposed:

```bash
python inference.py audio/ epoch-8-step-90000.ckpt embeddings/ \
    --segment-duration 20.0 \
    --overlap-ratio 0.5 \
    --batch-size 8 \
    --num-workers 6
```

One `.npy` file is written per input track, mirroring the input directory structure, containing that track's segment embeddings.

## Entry points

| Script | Purpose |
|---|---|
| `train.py` | Train a model from a YAML config |
| `validate.py` | Run the validation suite from a checkpoint |
| `inference.py` | Extract embeddings from audio |
| `evaluate.py` | Approximate retrieval + evaluation with cuVS IVF-Flat |
| `exhaustive-retrieval.py` | Brute-force retrieval against an embedding database |
| `manipulate-and-degrade.py` | Build manipulated/degraded query sets |
| `validate-from-ext-ti.py` | Track-ID evaluation from pre-extracted embeddings |
| `validate-from-ext-vi.py` | Version-ID evaluation from pre-extracted embeddings |

Every script documents its arguments under `--help`.

## Repository layout

```
src/common/          audio I/O, tensor ops, shared utilities
src/fish/            model, data pipeline, augmentations, validation
src/retrieval/       embedding databases, cuVS indices, track-level reduction
src/evaluation/      track- and version-identification metrics
src/signal_chain/    query manipulation and degradation (RIR, noise, codecs, ...)
src/similar_region/  similar-region and basin finding
configs/             training and signal-chain configurations
scripts/             preprocessing, similar-region, and SLURM job scripts
```

## Training data

The derived annotations are released as **Discogs-VI-SIREN** (SImilar REgioNs
between musical versions in Discogs-VI):

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21742034-blue)](https://doi.org/10.5281/zenodo.21742034)

<https://doi.org/10.5281/zenodo.21742034>

| File | Contents |
|---|---|
| `similar-regions.csv` | ~106 M rows — basins (matching regions) between pairs of versions |
| `segment-cliques.csv` | ~20.6 M rows — segments across versions covering the same musical passage |

Train and validation splits are separated by work. Point `train_dataloader.csv_path` (and the validation dataloaders) in `configs/train/fish.yaml` at the `segment-cliques.csv` for the corresponding split.

Downloading this record lets you skip the preprocessing, similar-region and segment-clique pipelines described below and go straight to training. The audio itself is not included — it comes from Discogs-VI-YT.

Discogs-VI-SIREN is released under the MIT license, separately from the code in this repository.

## Data preprocessing

### Discogs-VI

The bash scripts under `scripts/slurm/` record exactly how each step was run,
including the arguments used.

1. Convert to 16 kHz 16-bit wav — `scripts/slurm/preprocess-discogs-vi-yt.sh`
1. Silence filtering by RMS — `scripts/slurm/find-silent-tracks.sh`
1. Move the silent tracks to a separate directory
1. Tag the audio segments of the remaining tracks with
   [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn). We used the
   `Cnn14_16k` model (mAP 0.438) over 10 s windows with a 1 s hop, keeping the
   top 20 tags per window, with the 16 kHz mel front-end the checkpoint expects
   (window 512, hop 160, 64 mel bins, `fmin` 50, `fmax` 8000). It writes one
   JSON of per-segment tags per track, which is the input to the next step.
   Like the CLEWS step below, this runs someone else's model in its own
   environment. Our batched, SLURM-partitioned inference driver is a single
   commit on top of upstream, on the `unified-track-and-version-id` branch of
   [raraz15/audioset_tagging_cnn](https://github.com/raraz15/audioset_tagging_cnn/tree/unified-track-and-version-id)
1. Find which segments are non-music — `scripts/slurm/non-music-finder.sh`.
   This also reports tracks that are entirely silent
1. Move the completely non-music tracks to a separate directory
1. Split the remainder into `train/`, `val/database` and `test/database`
   (the clean tracks form the retrieval database)

## Similar region location

1. Extract CLEWS embeddings — `scripts/slurm/clews-embedding-extraction.sh`.
   This requires a checkout of the original [CLEWS](https://github.com/sony/clews)
   library. This step stands apart from the rest of the repository: it runs
   CLEWS, not our model, in its own conda environment with CLEWS's own
   dependencies — not the `unified` environment — and it is needed only to
   rebuild the training annotations from scratch. If you download the released
   segment-clique CSVs, you can skip it entirely.
1. Find non-music segments (see above)
1. Locate the top-10 similar regions —
   `scripts/slurm/similar-region-location.sh`

## Segment cliques

`scripts/slurm/segment-clique-location.sh` turns the pairwise similar regions into segment cliques, producing the `segment-cliques.csv` files that `train_dataloader.csv_path` and the validation dataloaders consume. These CSVs are exactly what is released as Discogs-VI-SIREN (see [Training data](#training-data)), so this pipeline only needs to be re-run if you want to rebuild them from scratch.

## Training

```bash
python train.py configs/train/fish.yaml
```

Paths in the config point at the authors' cluster and must be updated to your
own dataset locations. Weights & Biases logging is on by default and configured
under the `wandb:` key — pass `--no-wandb` to disable it, or change `entity` to
your own.

## Evaluation

```bash
python evaluate.py <query_embeddings> <ground_truth.csv> \
    --database-embeddings <db_dir> \
    --id-level track \
    --output-dir <results_dir>
```

`--id-level` selects between `track` and `version` identification. Reported
metrics include mean average precision (M-AP) and M-JNAR, the mean normalized
average ranking (unbiased variant).

## Datasets

The experiments use Discogs-VI and SHS100K2 for version identification, and the
neural-music-fp test set (drawn from FMA) for track identification. Query
degradation draws on TUT Acoustic Scenes 2016 for background noise, and the MIT
Survey, AIR and OpenAIR impulse response collections plus a microphone impulse
response set for convolutional degradation.

[`data/`](data/) holds everything needed to reproduce the evaluation except the
audio: the train/validation/test track lists in `data/splits/`, and the
ground-truth files `evaluate.py` consumes in `data/ground-truth/` — clique
definitions for version identification, and query-to-reference maps for track
identification. See [data/README.md](data/README.md).

## A note on the SLURM scripts

`scripts/slurm/` contains the exact job scripts used for this work. They are
included for transparency and to document how each stage was invoked. Partitions,
QOS names, module loads and dataset paths are specific to the authors' cluster,
so treat them as reference rather than as a portable interface — the Python
entry points they wrap are fully parameterised and are what you should build on.

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for
the full text.

The model components in `src/fish/model/nets/clews.py` are adapted from
[sony/clews](https://github.com/sony/clews), which is also MIT licensed; the
original copyright notice and permission notice are retained in that file.

The Discogs-VI-SIREN data released alongside this work carries its own MIT
license, stated on its [Zenodo record](https://doi.org/10.5281/zenodo.21742034).

## Citation

If you use this code, data, or model, please cite:

```bibtex
@inproceedings{araz_unified_2026,
title = {Unified {Music} {Identification} for {Tracks} and {Versions}},
booktitle = {Proc. of the 27th {Int}. {Soc}. for {Music} {Information} {Retrieval} {Conf}. ({ISMIR})},
author = {Araz, R. Oguz and Serrà, Joan and Mitsufuji, Yuki and Serra, Xavier and Bogdanov, Dmitry},
year = {2026},
}
```

> R. O. Araz, J. Serrà, Y. Mitsufuji, X. Serra, and D. Bogdanov, "Unified Music Identification for Tracks and Versions," in Proc. of the 27th Int. Soc. for Music Information Retrieval Conf. (ISMIR), 2026.
