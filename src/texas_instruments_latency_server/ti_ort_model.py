import re
import time
from pathlib import Path
from typing import Dict
from typing import Union

import numpy as np
import onnxruntime as ort


class TIOnnxruntimeModel:
    """Class for inference using ONNX runtime with Texas Instruments execution provider."""

    def __init__(self, model_path: Union[str, Path]):
        """
        Create a TIOnnxruntimeModel object.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to model. Path to file is interpreted as a path to onnx for CPU only inference, path to a directory is interpreted
            as a path to artifacts directory for NPU+CPU inference.

        """
        model_path = Path(model_path).resolve()

        if model_path.is_dir():
            providers = ["TIDLExecutionProvider", "CPUExecutionProvider"]
            ti_provider_options = {
                "tidl_tools_path": "",
                "artifacts_folder": str(model_path),
            }
            providers_options = [ti_provider_options, {}]

            onnx_path_variants = list(model_path.glob("*.onnx"))
            if len(onnx_path_variants) != 1:
                raise ValueError("Artifacts directory must contain only one onnx file.")

            model_path = onnx_path_variants[0]

        elif model_path.is_file():
            providers = ["CPUExecutionProvider"]
            providers_options = [{}]
        else:
            raise ValueError("model_path must be path to onnx file or artifacts directory")

        self.ort_session = ort.InferenceSession(
            str(model_path.resolve()),
            providers=providers,
            provider_options=providers_options,
        )

        self._dummy_input_feed = {}
        for input_info in self.ort_session.get_inputs():
            dtype = self._ort_value_dtype_to_np_dtype(input_info.type)
            self._dummy_input_feed[input_info.name] = np.ones(shape=input_info.shape, dtype=dtype)

    @property
    def batch_size(self) -> int:
        input_info = self.ort_session.get_inputs()[0]
        return input_info.shape[0]

    @staticmethod
    def _ort_value_dtype_to_np_dtype(ort_dtype: str) -> np.dtype:
        match_object = re.match(r"tensor\((.*)\)", ort_dtype)
        if match_object is None:
            raise ValueError(f"Got unknown tensor type '{ort_dtype}'")

        dtype = match_object.group(1)
        if dtype == "float":
            dtype = "float32"

        return np.dtype(dtype)

    def collect_ti_stats(self) -> Dict[str, float]:
        raw_stats = self.ort_session.get_TI_benchmark_data()  # pyright: ignore [reportGeneralTypeIssues]

        stats = {
            "total_ms": raw_stats["ts:run_end"] - raw_stats["ts:run_start"],
            "ddr_read_ms": raw_stats["ddr:read_end"] - raw_stats["ddr:read_start"],
            "ddr_write_ms": raw_stats["ddr:write_end"] - raw_stats["ddr:write_start"],
            "NPU_execution_ms": 0.0,
            "NPU_copy_input_ms": 0.0,
            "NPU_copy_output_ms": 0.0,
        }

        subgraph_ids = tuple(k[12:-11] for k in raw_stats if k.endswith("proc_start"))
        for subgraph_id in subgraph_ids:
            prefix = f"ts:subgraph_{subgraph_id}"
            stats["NPU_execution_ms"] += raw_stats[f"{prefix}_proc_end"] - raw_stats[f"{prefix}_proc_start"]
            stats["NPU_copy_input_ms"] += raw_stats[f"{prefix}_copy_in_end"] - raw_stats[f"{prefix}_copy_in_start"]
            stats["NPU_copy_output_ms"] += raw_stats[f"{prefix}_copy_out_end"] - raw_stats[f"{prefix}_copy_out_start"]

        stats = {k: v / 10**6 for k, v in stats.items()}

        stats["total_execution_ms"] = stats["total_ms"] - stats["NPU_copy_input_ms"] - stats["NPU_copy_output_ms"]
        stats["CPU_execution_ms"] = stats["total_execution_ms"] - stats["NPU_execution_ms"]

        return stats

    def benchmark_run(self) -> float:
        """
        Run inference one time with dummy input data and return time in ms.

        Returns
        -------
        float
            Time in ms.

        """
        t_0 = time.perf_counter()
        self.ort_session.run(output_names=None, input_feed=self._dummy_input_feed)
        return (time.perf_counter() - t_0) * 1000
