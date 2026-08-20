# Splits and ground truth

Everything needed to reproduce the paper's evaluation, apart from the audio
itself. Publishing these is what makes the numbers checkable: the splits were
decided once and cannot be re-derived from the preprocessing pipeline, and the
ground-truth files are what `evaluate.py` takes as its second positional
argument.

```
splits/          which tracks are in train / validation / test
ground-truth/    query -> reference correspondences for evaluation
```

## Splits

| File | Lines | Contents |
|---|---|---|
| `splits/discogs-vi-train-paths.txt` | 339,771 | Discogs-VI-YT training tracks |
| `splits/discogs-vi-val-paths.txt` | 37,081 | Discogs-VI-YT validation tracks |
| `splits/discogs-vi-test-paths.txt` | 116,197 | Discogs-VI-YT test tracks |
| `splits/shs100k-test-paths.txt` | 10,547 | SHS100K2 test tracks |
| `splits/nmfp-test-paths.txt` | 95,134 | neural-music-fp test tracks, as FMA source files |

One track per line, as a path relative to the root of the corresponding
preprocessed (16 kHz, 16-bit wav) audio collection:

```
kC/kCdwzJSdz-s.wav
```

The two-character parent directory is the first two characters of the YouTube
ID, matching the layout produced by `scripts/slurm/preprocess-discogs-vi-yt.sh`
and `scripts/slurm/preprocess-shs100k.sh`. Prefix them with your own audio root
to get absolute paths.

`nmfp-test-paths.txt` is the exception. The neural-music-fp test tracks come
from FMA, and we re-encode the *original* FMA files rather than upsampling the
8 kHz audio NMFP distributes, so this list gives paths relative to the root of
the FMA audio collection, with FMA's own extension:

```
000/000003.mp3
```

It is the input `scripts/slurm/preprocess-nmfp-test.sh` expects. The
preprocessed counterpart of each line is `test/database/000/000003.wav`, since
the conversion mirrors the FMA directory structure.

Two counts worth reconciling. Our run started from 95,163 FMA files, of which
29 failed to decode; those are already excluded here, so working from this list
reproduces our database exactly. And the NMFP track-identification ground truth
lists 94,684 queries against these 95,134 database tracks: 450 of them are
shorter than the 10 s query chunk, so they yield no query while still serving as
distractors in the database.

The Discogs-VI splits are separated by musical work, so no clique spans two
splits. The same partition underlies the segment-clique annotations released as
Discogs-VI-SIREN (<https://doi.org/10.5281/zenodo.21742033>).

## Version identification ground truth

`ground-truth/version-id/` holds clique definitions — which recordings are
versions of the same work. Pass one to `evaluate.py` with `--id-level version`.

| File | Format |
|---|---|
| `discogs-vi-test-cliques.json` | `{clique_id: [{version_id, track_title, youtube_id}, ...]}` |
| `shs100k2-test-cliques.json` | `{clique_id: [{version_id, youtube_id}, ...]}` |

`discogs-vi-test-cliques.json` is the test slice of the Discogs-VI-YT release
(`Discogs-VI-YT-20240701-light.json.test`), included here so the evaluation runs
without a separate download.

## Track identification ground truth

`ground-truth/track-id/` maps each degraded query chunk back to the database
recording it was cut from. Pass one to `evaluate.py` with `--id-level track`.

```
type,query_relative_path,reference_relative_path,chunk_boundary_start_idx,chunk_boundary_end_idx
chunks,chunks/clean/--/---YCQJodEE.wav,test/database/--/---YCQJodEE.wav,2025335,2185335
```

Boundary indices are sample offsets into the reference recording at 16 kHz.

| Directory | Query set |
|---|---|
| `discogs-vi-test/` | Discogs-VI-YT test queries |
| `neural-music-fp-test/` | neural-music-fp test queries (drawn from FMA) |

Each contains `clean.csv`, `clean-manipulated.csv` and
`clean-manipulated-degraded.csv`, one per query condition.

These six files are kept here so that a fresh clone can run `evaluate.py`
without a separate download. Byte-identical copies are also archived in the
evaluation query metadata record on Zenodo
(<https://doi.org/10.5281/zenodo.22027253>), which is the
canonical copy: it additionally carries the per-query manipulation and
degradation parameters, and the version-identification query metadata that is
not needed for evaluation and so is not duplicated here. If the two ever
disagree, the Zenodo record is correct.

## What is not here

The query audio itself. The manipulated and degraded queries are regenerated
from the clean test tracks with `manipulate-and-degrade.py`, which is seeded;
the CSVs above describe the result. Reproducing them does not have to rely on
the seed, though: the Zenodo record referenced above publishes the exact
parameters sampled for every individual query — which impulse responses, what
SNR, how much pitch shift — together with the degradation audio those
parameters refer to (<https://doi.org/10.5281/zenodo.22026053>).

Validation-split ground truth is also omitted — only the test sets are needed
to reproduce the reported results.

## AudioSet ontology

`audioset-ontology.json` is the AudioSet class hierarchy as an `anytree` tree.
`scripts/preprocess-audio/non-music-finder.py` reads it to expand the *Music*,
*Singing* and *Humming* families into the tag sets it treats as musical, and
uses it by default; override with `--ontology`.

The AudioSet ontology is published by Google under CC-BY-4.0:
<https://github.com/audioset/ontology>

The annotations that script produced over Discogs-VI-YT — which windows of which
recordings are non-music — are published as `nonmusic-annotations.tar.gz` in
Discogs-VI-SIREN (<https://doi.org/10.5281/zenodo.21742033>), so the tagging does
not have to be repeated. They cover the test split as well as train and val.
