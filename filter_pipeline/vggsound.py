from pathlib import Path
from multiprocessing import Pool, cpu_count

import tarfile

import pandas as pd
import huggingface_hub



DATA_DIR = Path("data") / "vggsound"
UNPACKED_DIR = DATA_DIR / "unpacked"

INDEX_FILE = "vggsound.csv"
SHARDS = [f"vggsound_{i:02d}.tar.gz" for i in range(20)]

def _build_file_index(folder: Path) -> pd.DataFrame:
    """Indexes all files in a given folder and maps them to """

    paths = folder.rglob("*.*")
    index_entries = [(path.stem, path.absolute()) for path in paths]
    df = pd.DataFrame.from_records(index_entries, columns=["file_id", "path"], index="file_id")

    return df


def _build_dataset_index(csv_file: Path) -> pd.DataFrame:
    """Read the csv file into a dataframe."""
    df = pd.read_csv(csv_file, names=["YouTube ID", "start seconds", "label", "train/test split"])
    df["file_id"] = df["YouTube ID"] + "_" + df["start seconds"].astype(str).str.zfill(6)
    return df


def _download_file(filename: str) -> Path:
    """Download file if it does not exist already."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / filename

    if path.exists():
        print(path, "already exists.")
    else:      
        huggingface_hub.hf_hub_download(
            repo_id="Loie/VGGSound",
            filename=filename,
            repo_type="dataset",
            local_dir=DATA_DIR
        )

    return path


def _unpack(args=tuple[Path, Path]):
    file, destination = args
    print(f"Unpacking {file} ...")
    with tarfile.open(file, "r:gz") as f:
        f.extractall(destination)


def _unpack_multiple(files: list[Path], destination: Path):
    args = [(file, destination) for file in files]
    with Pool(cpu_count() - 1) as p:
        p.map(_unpack, args)
        

def get_vggsound_dataset(n_shards: int=20) -> pd.DataFrame:
    """Download and organize the VGGSound dataset or a subset of it."""
    if n_shards > 20:
        print("Only 20 shards are available.")
        n_shards=20
    
    index_file = _download_file(INDEX_FILE)
    shards = [_download_file(shard) for shard in SHARDS[:n_shards]]

    print("Unpacking ...")
    _unpack_multiple(shards, UNPACKED_DIR)

    print("Preparing DataFrame")
    index_df = _build_dataset_index(index_file)
    files_df = _build_file_index(UNPACKED_DIR)
    matched_df = index_df.join(files_df, on="file_id", how="inner")

    return matched_df