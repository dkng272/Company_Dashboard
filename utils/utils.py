from pathlib import Path

def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parents[1]  # assuming utils/ is 1 level below root

def get_data_path(filename: str) -> Path:
    """Returns the full path to a file in the /data directory."""
    return get_project_root() / "data" / filename


