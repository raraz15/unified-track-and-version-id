#!/bin/bash
set -euo pipefail

SILENT_TXT_DIR="/home/raraz/unified-track-and-version-id/tmp/silent-tracks"

# SRC_WAV_DIR="/localdisk/raraz/unified-similarity/datasets/discogs-vi-yt-train-16kHz/train"
# DST_WAV_DIR="/localdisk/raraz/unified-similarity/datasets/discogs-vi-yt-train-16kHz-full-silent"
SRC_WAV_DIR="/localdisk/raraz/unified-similarity/datasets/discogs-vi-yt-test-val-16kHz/val/database"
DST_WAV_DIR="/localdisk/raraz/unified-similarity/datasets/full-silent/discogs-vi-yt-test-val-16kHz/val/database"

# mkdir -p "$DST_WAV_DIR"

# Loop through all text marker files
find "$SILENT_TXT_DIR" -type f -name "*.txt" | while read -r txtfile; do
    # Extract filename (ID)
    fname=$(basename "$txtfile" .txt)
    prefix=${fname:0:2}

    src="$SRC_WAV_DIR/$prefix/$fname.wav"

    if [[ -f "$src" ]]; then
        dst="$DST_WAV_DIR/$prefix/$fname.wav"
        mkdir -p "$(dirname "$dst")"
        echo "Moving $src → $dst"
        mv "$src" "$dst"
    else
        echo "WARNING: WAV not found for $fname"
    fi
done
