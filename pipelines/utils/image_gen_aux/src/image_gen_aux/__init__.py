# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from .utils import (
    IMAGE_AUX_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_transformers_available,
)


# Lazy Import based on
# https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py

# When adding a new object to this init, please add it to `_import_structure`. The `_import_structure` is a dictionary submodule to list of object names,
# and is used to defer the actual importing for when the objects are requested.
# This way `import image_gen_aux` provides the names in the namespace without actually importing anything (and especially none of the backends).

_import_structure = {
    "upscalers": [],
    "preprocessors": ["LineArtStandardPreprocessor"],
    "utils": [
        "OptionalDependencyNotAvailable",
        "is_torch_available",
        "logging",
    ],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    ...
else:
    _import_structure["upscalers"].extend(["UpscaleWithModel"])

    _import_structure["preprocessors"].extend(["DepthPreprocessor", "LineArtPreprocessor", "TeedPreprocessor"])

if TYPE_CHECKING or IMAGE_AUX_SLOW_IMPORT:
    from .preprocessors import LineArtStandardPreprocessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        ...
    else:
        from .preprocessors import LineArtPreprocessor, TeedPreprocessor
        from .upscalers import UpscaleWithModel

    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        ...
    else:
        from .preprocessors import DepthPreprocessor
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
