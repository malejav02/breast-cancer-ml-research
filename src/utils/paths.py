from pathlib import Path
from typing import Callable, Dict, Iterable, List, Union
from pyprojroot import here


def make_dir(dir_name: Union[str, Iterable[str]]) -> Callable[..., Path]:
    """
    Factory function that creates a directory path resolver relative to the project root.

    This function uses `pyprojroot.here()` to ensure that all paths are resolved
    from the root of the project, making the project structure portable and reproducible.

    Parameters
    ----------
    dir_name : Union[str, Iterable[str]]
        A directory name or a list/tuple of directory components.

    Returns
    -------
    Callable[..., Path]
        A function that builds full paths by appending subdirectories or filenames.

    """

    def dir_path(*args: str) -> Path:
        if isinstance(dir_name, str):
            return here().joinpath(dir_name, *args)
        return here().joinpath(*dir_name, *args)

    return dir_path


# Create the project directory function
project_dir = make_dir("")

dir_types: List[List[str]] = [
    ["data", "raw"],
    ["data", "processed"],
    ["models"],
    ["reports", "figures"],
    ["reports", "html"],
    ["docs"],
]

dir_functions: Dict[str, Callable[..., Path]] = {}

for dir_type in dir_types:
    DIR_VAR_NAME = "_".join(dir_type) + "_dir"
    dir_functions[DIR_VAR_NAME] = make_dir(dir_type)

data_raw_dir = dir_functions["data_raw_dir"]
data_processed_dir = dir_functions["data_processed_dir"]
models_dir = dir_functions["models_dir"]
reports_figures_dir = dir_functions["reports_figures_dir"]
reports_html_dir = dir_functions["reports_html_dir"]
