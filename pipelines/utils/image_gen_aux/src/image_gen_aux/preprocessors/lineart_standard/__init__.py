from typing import TYPE_CHECKING

from ...utils import IMAGE_AUX_SLOW_IMPORT, OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {}

try:
    if not (is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    ...
else:
    _import_structure["lineart_standard_preprocessor"] = [
        "LineArtStandardPreprocessor",
    ]

if TYPE_CHECKING or IMAGE_AUX_SLOW_IMPORT:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        ...
    else:
        from .lineart_standard_preprocessor import (
            LineArtStandardPreprocessor,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
