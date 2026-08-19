import json
from pathlib import Path
from typing import Union, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.common.audio import load_audio, SAMPLE_RATE


class ValRetrievalDataset(Dataset):
    def __init__(
        self,
        cliques_json_path: Union[str, Path],
        audio_dir: Union[str, Path],
        max_dur: Optional[float] = None,
        sample_rate: int = SAMPLE_RATE,
        clique_ratio: Optional[float] = None,
        verbose: bool = False,
    ) -> None:

        self.cliques_json_path = Path(cliques_json_path)
        self.audio_dir = Path(audio_dir)
        if clique_ratio is not None and not (0.0 < clique_ratio <= 1.0):
            raise ValueError(f"clique_ratio must be in (0, 1], got {clique_ratio}.")
        self.clique_ratio = clique_ratio
        self.sample_rate = sample_rate
        self.max_dur = max_dur
        self.verbose = verbose
        self.max_length = None
        if max_dur is not None:
            self.max_length = int(max_dur * sample_rate)
            if self.verbose:
                print(
                    f"[Retrieval] Maximum audio length set to {self.max_length:,} samples."
                )
        if self.verbose:
            print(f"[Retrieval] Loading cliques: \033[34m{cliques_json_path}\033[0m")
        with open(self.cliques_json_path) as f:
            self.cliques = json.load(f)
        self.count_cliques()

        # Delete versions with missing audio and add the audio path
        if self.verbose:
            print("[Retrieval] Deleting versions with missing audio...")
        for clique_id in list(self.cliques.keys()):
            delete = []
            for i in range(len(self.cliques[clique_id])):
                yt_id = self.cliques[clique_id][i]["youtube_id"]
                audio_path = self.audio_dir / yt_id[:2] / f"{yt_id}.wav"
                self.cliques[clique_id][i]["audio_path"] = audio_path
                if not audio_path.exists():
                    delete.append(i)
            for i in reversed(delete):
                del self.cliques[clique_id][i]
            # If a clique is left with less than 2 versions, delete the clique
            if len(self.cliques[clique_id]) < 2:
                del self.cliques[clique_id]
        self.count_cliques()

        # Optionally restrict to a deterministic subset of cliques
        if self.clique_ratio is not None and self.clique_ratio < 1.0:
            n_keep = max(1, round(self.clique_ratio * self.n_cliques))
            kept_ids = sorted(self.cliques.keys())[:n_keep]
            self.cliques = {k: self.cliques[k] for k in kept_ids}
            if self.verbose:
                print(
                    f"[Retrieval] Keeping {n_keep} cliques "
                    f"(clique_ratio={self.clique_ratio})."
                )
            self.count_cliques()

        # Create a list of all versions together with their clique ID
        self.items = []
        for clique_id, versions in self.cliques.items():
            for i in range(len(versions)):
                self.items.append((clique_id, i))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index) -> Tuple[torch.Tensor | None, int, int]:

        clique_id, version_idx = self.items[index]
        version_dict = self.cliques[clique_id][version_idx]
        clique_id = int(clique_id.split("C-")[1])
        version_id = int(version_dict["version_id"].split("V-")[1])
        audio_path = version_dict["audio_path"]

        # Load the entire audio clip (until max_length if set)
        audio = load_audio(audio_path, length=self.max_length)  # (1, T)

        return audio, clique_id, version_id

    def count_cliques(self):
        """Returns the number of cliques in the dataset."""

        # Count the number of cliques and versions
        self.n_cliques, self.n_versions = 0, 0
        for versions in self.cliques.values():
            self.n_cliques += 1
            self.n_versions += len(versions)
        if self.verbose:
            print(f" Cliques: {self.n_cliques:>7,}")
            print(f"Versions: {self.n_versions:>7,}")

    @staticmethod
    def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:

        audio_clips = []
        version_ids = []
        clique_ids = []
        lengths = []
        for audio, clique_id, version_id in batch:
            clique_ids.append(clique_id)
            version_ids.append(version_id)
            audio_clips.append(audio)
            lengths.append(audio.shape[1])  # return the original length
        clique_ids = torch.tensor(clique_ids)
        version_ids = torch.tensor(version_ids)

        # Zero pad all audio to the maximum length
        max_len = max(lengths)
        for i, (audio_clip, true_len) in enumerate(zip(audio_clips, lengths)):
            pad_right = max_len - true_len
            # NOTE: we remove these paddings later, so the mode is irrelevant
            audio_clips[i] = F.pad(
                audio_clip, (0, pad_right), mode="constant", value=0.0
            )
        audio_clips = torch.stack(audio_clips, 0).contiguous()  # (B,1,T_max)

        return audio_clips, clique_ids, version_ids, lengths
