"""Read trusted project archives made with NumPy 2 under NumPy 1.26.

Only serialized metadata module references are remapped; ZIPs and weights
remain untouched. This is a loading adapter, not a policy conversion.
"""
from __future__ import annotations

import base64
import io
import json
import pickle
from zipfile import ZipFile

import numpy as np
from gymnasium.spaces import Box


class ArchiveMetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_archived_model(algorithm_class, path):
    with ZipFile(path) as archive:
        metadata = json.loads(archive.read("data"))
    compatible = {}
    for key, value in metadata.items():
        if isinstance(value, dict) and ":serialized:" in value:
            if key in ("observation_space", "action_space"):
                # Recreate bounds/dtype, omitting an unused saved space RNG.
                # NumPy 2's Generator pickle state is incompatible with 1.26.
                if value[":type:"] != "<class 'gymnasium.spaces.box.Box'>":
                    raise ValueError("Only project Box spaces are supported")
                shape = tuple(value["_shape"])
                dtype = np.dtype(value["dtype"])
                bounds = [np.fromstring(value[name].strip("[]"), sep=" ", dtype=dtype)
                          for name in ("low", "high")]
                if any(bound.size != int(np.prod(shape)) for bound in bounds):
                    raise ValueError("Archive Box bounds do not match shape")
                compatible[key] = Box(bounds[0].reshape(shape), bounds[1].reshape(shape), dtype=dtype)
                continue
            payload = base64.b64decode(value[":serialized:"])
            compatible[key] = ArchiveMetadataUnpickler(io.BytesIO(payload)).load()
    return algorithm_class.load(path, device="cpu", custom_objects=compatible)
