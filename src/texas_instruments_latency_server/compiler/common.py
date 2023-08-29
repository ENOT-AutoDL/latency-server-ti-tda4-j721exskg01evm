import pickle
from pathlib import Path
from typing import Union

import numpy as np
import onnx
from onnx import helper as onnx_helper
from onnx import mapping as onnx_mapping


def _onnx_dtype_to_np_dtype(onnx_dtype: int) -> np.dtype:
    if hasattr(onnx_helper, "tensor_dtype_to_np_dtype"):
        return onnx_helper.tensor_dtype_to_np_dtype(onnx_dtype)

    # works in old versions of onnx lib
    return onnx_mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]


def generate_fake_calibration_data(model_path: Union[str, Path], output_dir: Path, n_output_files: int = 2) -> None:
    model = onnx.load(f=str(model_path), load_external_data=False)
    graph_inputs = model.graph.input

    for i in range(1, n_output_files + 1):
        fake_inputs = {}
        for graph_input in graph_inputs:
            shape = tuple(d.dim_value for d in graph_input.type.tensor_type.shape.dim)
            dtype = _onnx_dtype_to_np_dtype(graph_input.type.tensor_type.elem_type)
            fake_inputs[graph_input.name] = np.full(fill_value=i, shape=shape, dtype=dtype)

        with (output_dir / f"fake_calibration_data_{i}.pickle").open("wb") as output_file:
            pickle.dump(fake_inputs, file=output_file)
