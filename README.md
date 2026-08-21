# Unified Track and Version Identification

Official implementation of **"Unified Music Identification for Tracks and Versions"** (ISMIR 2026).

[![arXiv](https://img.shields.io/badge/arXiv-2608.19919-b31b1b)](https://arxiv.org/abs/2608.19919)

The pre-print is available at <https://arxiv.org/abs/2608.19919>.

A single embedding model handles both *track identification* (finding the exact recording a short, possibly degraded query came from) and *version identification* (finding other renditions of the same underlying musical work).
Audio is embedded segment-by-segment, and the same embedding space serves both retrieval tasks.

> **Note:** retrieval is GPU-only. The similarity search is built on [cuVS](https://github.com/rapidsai/cuvs) and there is no CPU fallback.
> **TODO:** Some ipynb notebooks for similar region and segment clique analysis, statistics, and visiualization are coming soon...

## Overview

| | |
|---|---|
| Input | 16 kHz mono audio, supports variable duration inputs but was trained with a 20s context window |
| Front-end | CQT — 12 bins/octave, 7 octaves from `fmin` 32.7 Hz, 20 ms hop |
| Encoder | CLEWS-style ResNet50-IBN with learnable GeM pooling |
| Embedding | 1024-d, L2-normalized |
| Training | Triplet loss with hard positive/negative mining, Adam + cosine annealing |
| Retrieval | cuVS IVF-Flat, or exhaustive search for smaller databases |

Training targets are built from *segment-level cliques*: rather than treating whole tracks as positives, the pipeline locates the regions of two versions that actually correspond, and trains on those.

## Why this codebase

Beyond reproducing the paper, some pieces here are worth reusing:

- **Track-level reduction on the GPU** — segment hits collapse to one score per track with no Python loop over candidates and no per-segment track-id array over the database: sort once, `bucketize` against track offsets, `scatter_reduce_`. [src/retrieval/reduction/](src/retrieval/reduction/)
- **Multi-GPU validation with proper sharding** — validation tracks are sharded across ranks and gathered back at their true lengths. No padding to a fixed segment count and no cutting tracks to a common length, because embeddings stay flat with a per-track `sizes` vector. [src/fish/model/litmodule.py](src/fish/model/litmodule.py)
- **cuVS instead of FAISS** — installs as an ordinary CUDA wheel with no build matrix to match against your toolkit, and takes torch tensors directly, so queries never leave the GPU. [src/retrieval/database/index.py](src/retrieval/database/index.py)
- **Approximate and exhaustive retrieval behind the same metrics** — `evaluate.py` and `exhaustive-retrieval.py`, so approximation error can be separated from model error.
- **Metrics without the usual bugs** — the relevant set counts only versions actually in the database, the query is dropped from its own results, truncating at K penalizes AP rather than inflating it, unretrieved relevant items are charged worst-case ranks in NAR, NAR is unbiased by default, and every number carries a confidence interval. [src/evaluation/metrics/](src/evaluation/metrics/)
- **Database tooling** — hundreds of thousands of per-track `.npy` files merge into one `float16` memmap plus a CSV of track offsets, built once and reused across runs; those same offsets drive the GPU reduction. [src/retrieval/database/](src/retrieval/database/)

> **TODO:** writing a trained index to disk and reloading it is implemented but disabled — a cuVS 25.8 bug with FP16 datasets, fixed in a later release. Until then every run retrains the index.

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

Selected at step 90000 as the maximum of the composite validation metric. It is a full Lightning checkpoint, so it can also be resumed from with `train.py --ckpt`.

## Quickstart: extracting embedding

Do *not* load the bare PyTorch model. The Lightning module's `extract_embeddings()` handles segmentation, padding and pooling the way the model expects, and `inference.py` drives it correctly — including multi-GPU.

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

The model was trained with 20s context duration but it supports variable length inputs. If an audio input is shorter than the specified segment duration, the code zero pads from the right. If the audio input is longer than the segment duration, the code segments the audio using the overlap ratio. So if you need to modify the inference segment duration, maybe read the paper to make an educated choice.

One `.npy` file is written per input track, mirroring the input directory structure, containing that track's segment embeddings.

## Entry points

| Script | Purpose |
|---|---|
| `train.py` | Train a model from a YAML config |
| `validate.py` | Run the validation suite from a checkpoint |
| `inference.py` | Extract embeddings from audio |
| `evaluate.py` | Approximate retrieval with cuVS IVF-Flat + evaluation (Main results in the paper) |
| `exhaustive-retrieval.py` | Exhaustive retrieval |
| `manipulate-and-degrade.py` | Build manipulated/degraded query sets |
| `validate-from-ext-ti.py` | Track-ID evaluation with Exhaustive retrieval from pre-extracted embeddings |
| `validate-from-ext-vi.py` | Version-ID evaluation with Exhaustive retrieval from pre-extracted embeddings |

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

The derived annotations are released as **Discogs-VI-SIREN** (SImilar REgioNs between musical versions in Discogs-VI):

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21742033-blue)](https://doi.org/10.5281/zenodo.21742033)

<https://doi.org/10.5281/zenodo.21742033>

| File | Contents |
|---|---|
| `similar-regions.csv` | ~106 M rows — basins (matching regions) between pairs of versions |
| `segment-cliques.csv` | ~20.6 M rows — segments across versions covering the same musical passage |
| `nonmusic-annotations.tar.gz` | 42,754 per-recording JSONs — the windows PANNs tagged as non-music, plus the 197 recordings that are non-music throughout |

Train and validation splits are separated by work. Point `train_dataloader.csv_path` (and the validation dataloaders) in `configs/train/fish.yaml` at the `segment-cliques.csv` for the corresponding split.

Downloading this record lets you skip the preprocessing, similar-region and segment-clique pipelines described below and go straight to training. It also carries the non-music annotations those pipelines consume, so the similar-region stage can be re-run without repeating the PANNs tagging — the one step that would otherwise mean running a second model over the whole collection. The audio itself is not included — it comes from Discogs-VI-YT.

Discogs-VI-SIREN is released under the MIT license, separately from the code in this repository.

## Data preprocessing

### Discogs-VI

We use Discogs-VI for training, and evaluation.

The bash scripts under `scripts/slurm/` record exactly how each step was run, including the arguments used.

1. Convert to 16 kHz 16-bit wav — `scripts/slurm/preprocess-discogs-vi-yt.sh`
1. Silence filtering by RMS — `scripts/slurm/find-silent-tracks.sh`
1. Move the silent tracks to a separate directory
1. Tag the audio segments of the remaining tracks with [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn). We used the `Cnn14_16k` model (mAP 0.438) over 10 s windows with a 1 s hop, keeping the top 20 tags per window, with the 16 kHz mel front-end the checkpoint expects (window 512, hop 160, 64 mel bins, `fmin` 50, `fmax` 8000). It writes one JSON of per-segment tags per track, which is the input to the next step. Like the CLEWS step below, this runs someone else's model in its own environment. Our batched, SLURM-partitioned inference driver is a single commit on top of upstream, on the `unified-track-and-version-id` branch of [raraz15/audioset_tagging_cnn](https://github.com/raraz15/audioset_tagging_cnn/tree/unified-track-and-version-id)
1. Find which segments are non-music — `scripts/slurm/non-music-finder.sh`. A window is non-music when its top PANNs tag falls outside the *Music*, *Singing* and *Humming* families of the AudioSet ontology. This also reports the tracks that are non-music throughout. **The output of steps 4 and 5 is published in Discogs-VI-SIREN as `nonmusic-annotations.tar.gz`, so both can be skipped**
1. Move the completely non-music tracks to a separate directory
1. Split the remainder into `train/`, `val/database` and `test/database` (the clean tracks form the retrieval database)

### Neural-Music-**FP**

We use the neural-music-fp test set for track identification only, so no filtering or splitting is needed — the tracks are used as the retrieval database as they are.

NMFP distributes its audio at 8 kHz, so instead of upsampling it we go back to the sources: the input is a line-delimited list of the *original* FMA files corresponding to the NMFP test tracks, released as [`data/splits/nmfp-test-paths.txt`](data/splits/nmfp-test-paths.txt) (95,134 tracks, as paths relative to the FMA audio root).

1. Convert those files to 16 kHz 16-bit mono wav — `scripts/slurm/preprocess-nmfp-test.sh`, which mirrors the FMA directory structure into `test/database/`

The same conversion applies to the degradation sets NMFP ships (background noise and impulse responses) — `scripts/slurm/preprocess-degradation.sh`, one collection per run. The resulting 16 kHz files are released on Zenodo:

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22026053-blue)](https://doi.org/10.5281/zenodo.22026053)

<https://doi.org/10.5281/zenodo.22026053>

### SHS100K2

We use the SHS100K2 test set for version identification only, again with no filtering or splitting.

1. Convert to 16 kHz 16-bit mono wav — `scripts/slurm/preprocess-shs100k.sh`. The audio comes from YouTube as `.mp4`, exactly like Discogs-VI-YT, so this reuses `scripts/preprocess-audio/preprocess-discogs-vi.py` unchanged

The track list is in [`data/splits/shs100k-test-paths.txt`](data/splits/shs100k-test-paths.txt).

## Similar region location

Discogs-VI groups versions at the level of whole tracks: it says that two recordings realise the same musical work, but not *where* in each recording the shared material sits.
Two versions can differ in intro length, section order, repeats, outros, and spoken or applause passages, so a 20 s window drawn at random from one is often not a positive for a 20 s window drawn at random from the other.
This stage recovers the missing alignment by locating, for every pair of versions inside a track clique, the regions of the two recordings that actually correspond.
We call one such matching region pair a *basin*; the basins are what `similar-regions.csv` contains.

The pipeline has three steps:

1. Extract CLEWS embeddings — `scripts/slurm/clews-embedding-extraction.sh`, wrapping `scripts/similar-region/clews-embedding-extraction.py`. This requires a checkout of the original [CLEWS](https://github.com/sony/clews) library. This step stands apart from the rest of the repository: it runs CLEWS, not our model, in its own conda environment with CLEWS's own dependencies — not the `unified` environment — and it is needed only to rebuild the training annotations from scratch. If you download the released segment-clique CSVs, you can skip it entirely.
1. Find non-music segments (see [Data preprocessing](#discogs-vi) above) — or download them as part of [Discogs-VI-SIREN](#training-data), which is what the released annotations were built with
1. Locate the similar regions — `scripts/slurm/similar-region-location.sh`, wrapping `scripts/similar-region/similar-region-location.py`

### CLEWS embeddings

Each track is cut into 20 s shingles with a 1 s hop, and every shingle is embedded independently with the released `dvi-clews` checkpoint.
The output is one `.npy` per track holding an `(n_shingles, d)` matrix in `float16`, mirroring the input directory layout, so a track becomes a *sequence* of embeddings rather than a single vector.
The driver runs under Lightning Fabric and shards the file list across GPUs with a `DistributedSampler`, skipping any track whose output already exists, so an interrupted job can simply be resubmitted.
Tracks shorter than 10 s are skipped.
The 1 s hop is what the rest of the pipeline is built around: it sets the time resolution at which a matching region can be localised, and it is the same hop the non-music annotations use, so the two index the same segment grid.

### Basin finding

`similar-region-location.py` walks the cliques of `Discogs-VI-YT-<date>-light.json` and processes each clique in its own worker process, caching a version's embeddings across all the pairs that version takes part in.
Cliques are sorted largest-first so the long-running ones start early and the worker pool drains evenly.

For each version, the CLEWS embedding matrix is loaded and then *masked*.
A shingle is discarded if it is silent — at least `--silence-min-fraction` (0.5) of its 0.1 s frames fall below `--silence-threshold-db` (−30 dBFS) — or if the PANNs step tagged it as non-music.
Masking sets those embeddings to `inf`, which makes every distance involving them infinite and so removes them from consideration without disturbing the segment indexing.
A version with 50 % or more silent shingles, or 50 % or more non-music shingles, is dropped from the clique entirely.

For each pair of surviving versions, a full pairwise distance matrix `H` is computed between their shingle embeddings, using the dimension-normalized squared Euclidean distance.
`H[i, j]` is the distance between the shingle starting at second `i` of the first version and the shingle starting at second `j` of the second, so `H` is a cross-similarity image in which corresponding passages show up as low-valued regions.
The pair is skipped if masking left the matrix entirely non-finite.

`find_basins` in [src/similar_region/basin_finding.py](src/similar_region/basin_finding.py) then extracts matching regions from `H` greedily:

1. Take the global minimum of the matrix — the *hole* — which is the best-matching shingle pair still available.
1. Stop if that hole is not deep enough, i.e. its distance is at or above the ridge height. Holes are consumed in increasing order of depth, so the first failure ends the search for this pair.
1. Flood-fill from the hole with a tolerance of `ridge_height − hole_height`, using 4-connectivity. The flooded region is the *basin*: the contiguous set of cells around the hole that stay similar, which is what distinguishes a passage that matches over time from a single lucky frame.
1. Mask the basin's bounding rectangle to `inf` (`--removal-type rectangle`, the setting we used) and repeat, forcing the next iteration into a different part of the matrix.

The ridge height is `min(--global-pct-threshold, --universal-dist-threshold)`: the 5th percentile of the finite distances in *this* matrix, capped at an absolute distance of 2.0.
The percentile term adapts to the pair — a close pair has many low cells, a distant pair few — while the absolute cap stops a pair with no genuine correspondence from having its own 5th percentile accepted as a match, which is the main source of false positives.
Two shape filters follow: a basin of area 1 is a single-cell artifact and is dropped, and a basin whose bounding box covers 900 cells or more is dropped as too diffuse to name a specific passage.
At most `--max-basins-write` (50) basins are written per version pair.

The output is one `similar-regions.csv` per split, one row per basin, with the columns listed in [src/similar_region/field_names.py](src/similar_region/field_names.py): the track clique id, both version and YouTube ids, both track durations, the shingle duration, the basin index `k` within the pair, the hole depth `hole_height`, the hole's start time in each version (`hole_v0_start`, `hole_v1_start`, in seconds), and `basin_area`.
A row therefore asserts that `[hole_v0_start, hole_v0_start + segment_duration)` in the first version and `[hole_v1_start, hole_v1_start + segment_duration)` in the second cover the same musical passage.
Rows are flushed per clique, and a `parameters-similar-regions.json` recording every argument, the run timestamp and the git commit is written next to the CSV.

## Segment cliques

The similar regions are strictly *pairwise*: each row relates one segment of one version to one segment of one other version.
Training with hard positive mining needs more than that — given an anchor segment, we want every other segment covering the same passage, across all versions of the work.
`scripts/slurm/segment-clique-location.sh`, wrapping `scripts/similar-region/segment-clique-location.py`, computes that transitive closure and turns pairwise basins into *segment cliques*.

Each basin row is expanded into its two 20 s intervals, one per version.
Within a track clique all intervals are bucketed by `version_id`, and two basins are merged when they place intervals on the *same* version that overlap by at least `--merge-condition-dur` (19 s out of 20).
Merging uses a union-find, so the relation is transitive: if basins A and B nearly coincide on version *x*, and B and C nearly coincide on version *y*, then A, B and C all land in one component even though A and C were never compared directly.
Each connected component becomes one segment clique, holding the intervals of every basin in it.
The 19 s threshold is deliberately strict — it demands near-identical intervals rather than mere overlap, because a loose threshold lets a component chain along slightly shifted windows and eventually swallow the whole track.
Identical `(version_id, start, end)` triples inside a clique are de-duplicated, keeping the first occurrence.

The result is `segment-cliques.csv`, one row per segment, with `track_clique_id`, `segment_clique_id` (a running `S-0000000` identifier), `version_id`, `youtube_id`, `segment_start_time`, `segment_end_time` and `track_duration`.
Rows sharing a `segment_clique_id` are mutually positive at the segment level, and this is exactly what `train_dataloader.csv_path` and the validation dataloaders consume.
A `parameters-segment-cliques.json` is written alongside, again recording the arguments, timestamp and git commit.

These CSVs are exactly what is released as Discogs-VI-SIREN (see [Training data](#training-data)), so this pipeline only needs to be re-run if you want to rebuild them from scratch.

## Training

```bash
python train.py configs/train/fish.yaml
```

The model reported in the paper was trained on 4 GPUs. `scripts/slurm/train-4-gpu.sh` and `scripts/slurm/train-1-gpu.sh` are the SLURM job scripts used for the 4-GPU and single-GPU cases respectively; the 4-GPU script launches `train.py` under `torchrun` for distributed data parallel training, the 1-GPU script calls `train.py` directly.

Paths in the config point at the authors' cluster and must be updated to your own dataset locations. Weights & Biases logging is on by default and configured under the `wandb:` key — pass `--no-wandb` to disable it, or change `entity` to your own.

## Evaluation

```bash
python evaluate.py <query_embeddings> <ground_truth.csv> \
    --database-embeddings <db_dir> \
    --id-level track \
    --output-dir <results_dir>
```

`--id-level` selects between `track` and `version` identification. Reported metrics include mean average precision (M-AP) and M-NAR, the mean normalized average ranking (unbiased variant).

The database must be given in exactly one of two forms: `--database-embeddings` for a directory of per-track `.npy` files, or `--database-memmap` for a `database.mm` built by an earlier run. Either way the IVF-Flat index is trained at the start of the run; see the TODO under [Why this codebase](#why-this-codebase).

### SLURM wrappers

`scripts/slurm/` holds the job scripts the reported numbers were produced with. The per-task ones fix whatever differs between the two tasks and take the paths positionally:

```bash
# queries  ground-truth  output-dir  db-flag  db-path
sbatch scripts/slurm/evaluate-track.sh   <queries> <gt.csv> <out> --database-embeddings <db_dir>
sbatch scripts/slurm/evaluate-version.sh <queries> <gt>     <out> --database-memmap     <db.mm>
```

`evaluate-track.sh` runs `--id-level track` with `--top-N 10`, `evaluate-version.sh` runs `--id-level version` with `--top-N 10000`, and both set `--num-workers` from the job's CPU allocation. `scripts/slurm/evaluate.sh` is the unconstrained variant: it also takes `--n-lists`, `--n-probes`, `--top-k` and `--top-N` as arguments, which is how the cuVS parameters were swept. `scripts/slurm/exhaustive-retrieval.sh` just forwards `"$@"` to `exhaustive-retrieval.py`.

### The `-all-queries` scripts

`evaluate-track-all-queries.sh` and `evaluate-version-all-queries.sh` are the ones we actually launch. Given a single embedding root, each loops over the three query conditions — `clean`, `clean-manipulated`, `clean-manipulated-degraded` — deriving the query, database, ground-truth and output paths itself, and choosing `--database-memmap` over `--database-embeddings` automatically when a `database.mm` is present.

```bash
sbatch scripts/slurm/evaluate-track-all-queries.sh   <emb_root> <audio_root> [query_type_idx]
sbatch scripts/slurm/evaluate-version-all-queries.sh <emb_root> [query_type_idx]
```

The optional trailing index (`0`, `1` or `2`) restricts the run to one query condition instead of all three.

They assume the layout that `inference.py` produces over the query sets built by `manipulate-and-degrade.py`:

```
<emb_root>/database/                     reference embeddings (plus database.mm if built)
<emb_root>/queries/chunks/<query_type>/  track-ID queries (short excerpts)
<emb_root>/queries/tracks/<query_type>/  version-ID queries (full tracks)
```

Results go to `<emb_root>/../../eval/track-id/<dataset>/chunks/<query_type>/` and `<emb_root>/../../eval/version-id/<dataset>/<query_type>/`, with `<dataset>` the basename of the embedding root. The two differ in where the ground truth comes from: the track-ID script reads the `ground-truth.csv` that `manipulate-and-degrade.py` wrote next to the query audio — hence the extra `<audio_root>` argument — while the version-ID script picks the Discogs-VI or SHS100K2 clique file by matching `dvi` or `shs` in the embedding root's name. For version identification the `clean` queries are the database tracks themselves, so that condition queries the database against itself.

Both are cluster-specific in their partitions and, for version identification, in their hard-coded ground-truth paths — see [A note on the SLURM scripts](#a-note-on-the-slurm-scripts).

## Datasets

The experiments use [Discogs-VI](https://doi.org/10.5281/zenodo.13983028) for both version identification and track identification, [SHS100K2](https://github.com/NovaFrost/SHS100K2) for version identification, and the [neural-music-fp](https://github.com/raraz15/neural-music-fp#dataset-for-training-and-evaluation) test set (drawn from FMA) for track identification. Query degradation draws on TUT Acoustic Scenes 2016 for background noise, and the MIT Survey, AIR and OpenAIR impulse response collections plus a microphone impulse response set for convolutional degradation. All of these degradation sets are taken from the neural-music-fp (NMFP) data release. NMFP distributes its audio at 8 kHz, whereas we work at 16 kHz, so we re-processed the original source files (44.1 kHz and above) to 16 kHz — the audio files themselves are the same, only the sample rate differs. The 16 kHz versions of the NMFP degradation sets are released at <https://doi.org/10.5281/zenodo.22026053>.

[`data/`](data/) holds everything needed to reproduce the evaluation except the audio: the train/validation/test track lists in `data/splits/`, and the ground-truth files `evaluate.py` consumes in `data/ground-truth/` — clique definitions for version identification, and query-to-reference maps for track identification. See [data/README.md](data/README.md).

The query audio is not released, but the metadata needed to rebuild it is. The evaluation query metadata record holds, for every track- and version-identification query across all three test sets, the chunk timestamps and the exact parameters `manipulate-and-degrade.py` sampled for it — which impulse responses, what SNR, how much pitch shift:

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22027253-blue)](https://doi.org/10.5281/zenodo.22027253)

<https://doi.org/10.5281/zenodo.22027253>

## A note on the SLURM scripts

`scripts/slurm/` contains the exact job scripts used for this work. They are included for transparency and to document how each stage was invoked. Partitions, QOS names, module loads and dataset paths are specific to the authors' cluster, so treat them as reference rather than as a portable interface — the Python entry points they wrap are fully parameterised and are what you should build on.

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for the full text.

The model components in `src/fish/model/nets/clews.py` are adapted from [sony/clews](https://github.com/sony/clews), which is also MIT licensed; the original copyright notice and permission notice are retained in that file.

The Discogs-VI-SIREN data released alongside this work carries its own MIT license, stated on its [Zenodo record](https://doi.org/10.5281/zenodo.21742033).

## Citation

If you use this code, data, or model, please cite:

```bibtex
@inproceedings{araz_unified_2026,
title = {Unified {Music} {Identification} for {Tracks} and {Versions}},
booktitle = {Proc. of the 27th {Int}. {Soc}. for {Music} {Information} {Retrieval} {Conf}. ({ISMIR})},
author = {Araz, R. Oguz and Serrà, Joan and Mitsufuji, Yuki and Serra, Xavier and Bogdanov, Dmitry},
year = {2026},
eprint = {2608.19919},
archiveprefix = {arXiv},
primaryclass = {cs.SD},
url = {https://arxiv.org/abs/2608.19919},
}
```

> R. O. Araz, J. Serrà, Y. Mitsufuji, X. Serra, and D. Bogdanov, "Unified Music Identification for Tracks and Versions," in Proc. of the 27th Int. Soc. for Music Information Retrieval Conf. (ISMIR), 2026. arXiv:2608.19919.
