#!/bin/bash

set -e

ENV_NAME="${ENV_NAME:-unified}"

echo "Creating conda environment '$ENV_NAME'"
conda create -n $ENV_NAME python=3.11 -y

# ffmpeg is installed with conda because torchaudio needs a system-level build,
# which pip cannot provide.
echo "Installing ipython and ffmpeg with conda"
conda run -n $ENV_NAME conda install -c anaconda ipython -y
conda run -n $ENV_NAME conda install -c conda-forge 'ffmpeg<7' -y

# The dependency list lives in pyproject.toml; this only supplies the extra
# indexes that the CUDA and cuVS wheels are published on.
echo "Installing the project and its dependencies with pip"
conda run -n $ENV_NAME pip install -e ".[gpu,train]" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.nvidia.com

echo "Verifying torchaudio audio backends in '$ENV_NAME'..."
if conda run -n $ENV_NAME python -c "import torchaudio; backends = torchaudio.list_audio_backends(); exit(0) if 'ffmpeg' in backends and 'soundfile' in backends else exit(1)"; then
    echo "✅ Torchaudio backends ('ffmpeg', 'soundfile') are available."
else
    echo "❌ Torchaudio did not find the expected audio backends." >&2
fi

echo "Installation complete! To use the environment, run: conda activate $ENV_NAME"
