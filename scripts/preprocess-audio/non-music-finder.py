"""The methodology here for building the sound class families were described in
https://ieeexplore.ieee.org/abstract/document/10859250
The code is available at:
https://github.com/raraz15/semantic-sound-similarity
"""


from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import argparse

from anytree import PreOrderIter
from anytree.importer import JsonImporter
from anytree.search import find_by_attr


DEFAULT_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "audioset-ontology.json"
)


def load_ontology(ontology_path: Path = DEFAULT_ONTOLOGY_PATH):
    importer = JsonImporter()
    with open(ontology_path, "r") as infile:
        tree_dict = json.load(infile)
    root = importer.import_(json.dumps(tree_dict))
    return root


def find_node(root, name):
    node = find_by_attr(root, name, name="name")
    if node is None:
        raise RuntimeError(f"Node '{name}' not found.")
    return node


def build_family(root, family_root_raw_name):
    """Return the set of tags under a given node (including itself)."""
    family_root = find_node(root, family_root_raw_name)
    return {node.name for node in PreOrderIter(family_root)}


def process_json_file(task):
    json_path, output_path_json, output_path_txt, interested_tags, k = task
    with open(json_path) as in_f:
        predictions = json.load(in_f)

    bad_segments_dict = {}
    # Process each segment independently
    for seg_idx, seg_preds in predictions.items():
        # Get the set of tags in the top-k predictions
        segment_tags = {pred["label"] for pred in seg_preds[:k]}
        # If none of the tags are in the interested set, mark this segment as bad
        if not (segment_tags & interested_tags):
            # Store the top-k predictions and their probabilities for this bad segment
            bad_segments_dict[seg_idx] = seg_preds[:k]

    if bad_segments_dict:
        output_path_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_json, "w") as out_f:
            json.dump(bad_segments_dict, out_f)

        # If all segments are non-music mark the file
        if len(bad_segments_dict) == len(predictions):
            print(
                f"{len(bad_segments_dict)} / {len(predictions)} segments in {json_path} are non-music."
            )
            output_path_txt.parent.mkdir(parents=True, exist_ok=True)
            output_path_txt.write_text(
                f"{len(bad_segments_dict)} / {len(predictions)} segments are non-music.\n"
            )

    return bad_segments_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "json_dir", type=Path, help="Directory containing JSON files to scan"
    )
    parser.add_argument(
        "output_dir_json",
        type=Path,
        help="Directory to save non-music segments info in JSON files",
    )
    parser.add_argument(
        "output_dir_txt",
        type=Path,
        help="Directory to save tracks with all non-music segments in TXT files",
    )
    parser.add_argument(
        "--k", type=int, default=1, help="Number of top predictions to consider"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (defaults to CPU count)",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=DEFAULT_ONTOLOGY_PATH,
        help="Path to the AudioSet ontology tree JSON.",
    )
    args = parser.parse_args()

    json_paths = list(args.json_dir.rglob("*.json"))
    print(f"Found {len(json_paths)} JSON files.")

    ontology_root = load_ontology(args.ontology)
    music_tags = build_family(ontology_root, "Music")
    singing_tags = build_family(ontology_root, "Singing")
    humming_tags = build_family(ontology_root, "Humming")
    interested_tags = music_tags | singing_tags | humming_tags
    print(
        f"Identified {len(interested_tags)} interested tags (music, singing, humming)."
    )

    # Precompute all input/output path tuples and create directories once on main process
    tasks = []
    for json_path in json_paths:
        rel_path = json_path.relative_to(args.json_dir)
        output_path_json = args.output_dir_json / rel_path
        output_path_txt = args.output_dir_txt / rel_path.with_suffix(".txt")
        tasks.append(
            (json_path, output_path_json, output_path_txt, interested_tags, args.k)
        )

    bad_tracks = 0
    bad_segments = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(process_json_file, tasks)
        for (json_path, _, _, _, _), bad_segments_dict in zip(tasks, results):
            if bad_segments_dict:
                bad_tracks += 1
                bad_segments += len(bad_segments_dict)
    print(f"Total tracks with at least one bad segment: {bad_tracks:,}")
    print(f"Total amount of bad segments: {bad_segments:,}")
    print("Done.")
