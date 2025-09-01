from typing import TYPE_CHECKING

from ..utils import (
    IMAGE_AUX_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_transformers_available,
)


_import_structure = {
    "lineart": [],
    "depth": [],
    "teed": [],
    "lineart_standard": ["LineArtStandardPreprocessor"],
}


try:
    if not (is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    ...
else:
    _import_structure["lineart"] = ["LineArtPreprocessor"]
    _import_structure["teed"] = ["TeedPreprocessor"]

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    ...
else:
    _import_structure["depth"] = [
        "DepthPreprocessor",
    ]

if TYPE_CHECKING or IMAGE_AUX_SLOW_IMPORT:
    from .lineart_standard import LineArtStandardPreprocessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        ...
    else:
        from .lineart import LineArtPreprocessor
        from .teed import TeedPreprocessor

    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        ...
    else:
        from .depth import DepthPreprocessor
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
