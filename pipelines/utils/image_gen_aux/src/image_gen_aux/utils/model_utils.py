import os

from huggingface_hub import hf_hub_download, model_info


def get_model_path(pretrained_model_or_path, filename=None, subfolder=None):
    """
    Retrieves the path to the model file.

    If `pretrained_model_or_path` is a file, it returns the path directly.
    Otherwise, it attempts to find a `.safetensors` file associated with the given model path.
    If no `.safetensors` file is found, it raises a `FileNotFoundError`.

    Parameters:
    - pretrained_model_or_path (str): Path to the pretrained model or directory containing the model.
    - filename (str, optional): Specific filename to load. If not provided, the function will search for a `.safetensors` file.
    - subfolder (str, optional): Subfolder within the model directory to look for the file.

    Returns:
    - str: Path to the model file.

    Raises:
    - FileNotFoundError: If no `.safetensors` file is found when `filename` is not provided.
    """
    if os.path.isfile(pretrained_model_or_path):
        return pretrained_model_or_path

    if filename is None:
        # If the filename is not passed, we only try to load a safetensor
        info = model_info(pretrained_model_or_path)
        filename = next(
            (sibling.rfilename for sibling in info.siblings if sibling.rfilename.endswith(".safetensors")), None
        )
        if filename is None:
            raise FileNotFoundError("No safetensors checkpoint found.")

    return hf_hub_download(pretrained_model_or_path, filename, subfolder=subfolder)
