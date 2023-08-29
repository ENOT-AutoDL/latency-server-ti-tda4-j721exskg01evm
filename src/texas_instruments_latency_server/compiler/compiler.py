import logging
import os
import pickle
import pprint
import shutil
from pathlib import Path
from typing import Optional
from typing import Union

import onnxruntime as ort
from onnx import shape_inference

from texas_instruments_latency_server.compiler.config import TICalibrationCfg
from texas_instruments_latency_server.compiler.config import TIDebugLevel
from texas_instruments_latency_server.compiler.config import TIModelCfg
from texas_instruments_latency_server.compiler.config import TIPrecisionCfg

_LOGGER = logging.getLogger(__name__)


class TICompiler:
    def __init__(
        self,
        tidl_tools_path: Optional[Union[str, Path]] = None,
        debug_level: TIDebugLevel = TIDebugLevel.NO_DEBUG,
        max_num_subgraphs: int = 16,
        ti_internal_nc_flag: int = 0,
    ):
        if tidl_tools_path is None:
            try:
                tidl_tools_path = os.environ["TIDL_TOOLS_PATH"]
            except KeyError as exc:
                raise ValueError(
                    "tidl_tools_path must be set manually by user as "
                    "a compiler option or as an env variable TIDL_TOOLS_PATH"
                ) from exc

        self._tidl_tools_path = Path(tidl_tools_path).resolve()
        self.check_compiler()
        self.debug_level = debug_level

        if max_num_subgraphs <= 0 or max_num_subgraphs > 16:
            raise ValueError("max_num_subgraphs must be in the range (0, 16]")

        self._max_num_subgraphs = max_num_subgraphs
        self.ti_internal_nc_flag = ti_internal_nc_flag

    @property
    def max_num_subgraphs(self) -> int:
        return self._max_num_subgraphs

    @staticmethod
    def check_compiler():
        available_providers = ort.get_available_providers()
        _LOGGER.info(f"available_providers: {available_providers}")
        if "TIDLCompilationProvider" not in available_providers:
            raise RuntimeError("Compiler provider is not available.")

    @staticmethod
    def _run_calibration(ort_session: ort.InferenceSession, calibration_data_dir_path: Union[str, Path]) -> None:
        _LOGGER.info("run calibration")

        calibration_data_dir_path = Path(calibration_data_dir_path)
        for input_data_path in calibration_data_dir_path.glob("*.pickle"):
            with input_data_path.open("rb") as input_data_file:
                input_data = pickle.load(input_data_file)
                ort_session.run(output_names=None, input_feed=input_data)

        _LOGGER.info("calibration finished")

    def compile(
        self,
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        calibration_data_dir: Union[str, Path],
        model_cfg: TIModelCfg,
        precision_cfg: TIPrecisionCfg,
        calibration_cfg: TICalibrationCfg,
        copy_onnx_to_output_dir: bool = True,
        disable_shape_inference: bool = False,
        force_overwrite: bool = True,
    ) -> None:
        output_dir = Path(output_dir).resolve()
        if output_dir.exists():
            if not output_dir.is_dir():
                raise ValueError(f"output_dir_path ('{output_dir}') must be path to a directory")

            if force_overwrite:
                shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True)
            else:
                raise ValueError(f"output directory ('{output_dir}') already exists")
        else:
            output_dir.mkdir(parents=True)

        compiler_options = {
            "platform": "J7",
            "version": "7.2",
            "debug_level": self.debug_level.value,
            "tidl_tools_path": str(self._tidl_tools_path),
            "artifacts_folder": str(output_dir),
            "max_num_subgraphs": self.max_num_subgraphs,
            "ti_internal_nc_flag": self.ti_internal_nc_flag,
        }
        compiler_options.update(model_cfg.as_cfg_dict())
        compiler_options.update(precision_cfg.as_cfg_dict())
        compiler_options.update(calibration_cfg.as_cfg_dict())

        calibration_data_dir = Path(calibration_data_dir)
        if not calibration_data_dir.is_dir() or not calibration_data_dir.exists():
            raise ValueError(f"directory with calibration data is not found ('{calibration_data_dir}')")
        calibration_frames = sum(1 for _ in calibration_data_dir.glob("*.pickle"))
        if calibration_frames == 0:
            raise ValueError("cannot find any calibration input ('*.pickle')")

        compiler_options.update({"advanced_options:calibration_frames": calibration_frames})

        model_path = Path(model_path).resolve()
        if not disable_shape_inference:
            _LOGGER.info("start shape inference")
            shape_inference.infer_shapes_path(model_path=str(model_path), output_path=str(model_path))

        _LOGGER.info(f"final compiler options:\n{pprint.pformat(compiler_options)}")
        ort_session = ort.InferenceSession(
            path_or_bytes=str(model_path),
            providers=["TIDLCompilationProvider", "CPUExecutionProvider"],
            provider_options=[compiler_options, {}],
        )
        self._run_calibration(ort_session=ort_session, calibration_data_dir_path=calibration_data_dir)

        if copy_onnx_to_output_dir:
            shutil.copy(src=str(model_path), dst=str(output_dir / model_path.name))
