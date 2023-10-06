import argparse
import io
import multiprocessing
import pickle
import shutil
import time
import zipfile
from functools import partial
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Union

import onnx
from aiohttp import web
from aiohttp.web import HTTPInternalServerError
from enot_latency_server.client import measure_latency_remote
from enot_latency_server.server import LatencyServer

from texas_instruments_latency_server.compiler.common import generate_fake_calibration_data
from texas_instruments_latency_server.compiler.compiler import TICompiler
from texas_instruments_latency_server.compiler.config import TIAccuracyLevel
from texas_instruments_latency_server.compiler.config import TICalibrationCfg
from texas_instruments_latency_server.compiler.config import TIDebugLevel
from texas_instruments_latency_server.compiler.config import TIModelCfg
from texas_instruments_latency_server.compiler.config import TIPrecisionCfg
from texas_instruments_latency_server.compiler.config import TITensorBits


class MainLatencyServer(LatencyServer):
    def __init__(
        self,
        ti_host: str,
        ti_port: int,
        host: str = "0.0.0.0",
        port: int = 15003,
        tensor_bits: TITensorBits = TITensorBits.TENSOR_8_BITS,
        working_dir: Union[str, Path] = "./working_dir",
    ):
        """
        This server receives an ONNX model, compiles it and sends it to remote server on TI to measure model latency.

        Parameters
        ----------
        ti_host : str
            Host name or IP address of server on TI.
        ti_port : int
            Port of server on TI.
        host : str
            Host name or IP address of CompilationServer. Default value is '0.0.0.0'.
        port : int
            Port of CompilationServer. Default value is 15003.
        tensor_bits : TITensorBits
            Tensors bitness. 8 by default.
        working_dir : Path
            Working directory for tmp files.

        """
        super().__init__(host=host, port=port)

        self.ti_host = ti_host
        self.ti_port = ti_port

        self._tensor_bits = tensor_bits
        self._working_dir = Path(working_dir).resolve()
        self._model_path = self._working_dir / "model.onnx"
        self._artifacts_dir = self._working_dir / "artifacts"
        self._calibration_data_dir = self._working_dir / "calibration_data"

    def run(self) -> None:
        async def _compile_model_handler(request: web.Request) -> web.FileResponse:
            data: Dict = pickle.loads(await request.read())
            path_to_compiled_model: Path = await self._loop.run_in_executor(
                self._executor, partial(self._compile_model, **data)
            )
            return web.FileResponse(path_to_compiled_model, status=200)

        self._app.add_routes([web.post(path="/compile", handler=_compile_model_handler)])
        super().run()

    def _cleanup_working_dir(self) -> None:
        if self._working_dir.exists():
            shutil.rmtree(str(self._working_dir))

        self._working_dir.mkdir(parents=True)
        self._artifacts_dir.mkdir()
        self._calibration_data_dir.mkdir()

    @staticmethod
    def _run_compilation(*args, **kwargs) -> None:
        compiler = TICompiler(debug_level=TIDebugLevel.NO_DEBUG)
        compiler.compile(*args, **kwargs)

    def _run_isolated_compilation(self, *args, **kwargs) -> None:
        with multiprocessing.Pool(processes=1) as pool:
            pool.apply(self._run_compilation, args=args, kwds=kwargs)

    def _compile_model(self, model: bytes, calibration_data: Optional[bytes] = None) -> Path:
        onnx_model = onnx.load_model_from_string(model)
        onnx.save(onnx_model, f=str(self._model_path))

        print("Start compilation!")
        start = time.time()
        # TODO: we need to pass model type (classification, detection, etc.), number of classes, name of input node, ...
        # TODO: now we rely on hardcode and only can measure a classification model
        try:
            if calibration_data is None:
                # generate minimal data for calibration
                generate_fake_calibration_data(
                    model_path=self._model_path,
                    output_dir=self._calibration_data_dir,
                    n_output_files=2,
                )
                calibration_cfg = TICalibrationCfg(
                    accuracy_level=TIAccuracyLevel.BASIC,
                    calibration_iterations=1,
                )
            else:
                calibration_data_zip = io.BytesIO(calibration_data)
                if zipfile.is_zipfile(calibration_data_zip):
                    with zipfile.ZipFile(calibration_data_zip, mode="r") as zf:
                        zf.extractall(self._calibration_data_dir)
                else:
                    raise ValueError("Calibration data must be a zip file")

                calibration_cfg = TICalibrationCfg(
                    accuracy_level=TIAccuracyLevel.ADVANCED,
                    calibration_iterations=10,
                    pre_batchnorm_fold=True,
                    activation_clipping=True,
                    weight_clipping=True,
                    bias_calibration=True,
                )

            self._run_isolated_compilation(
                model_path=self._model_path,
                output_dir=self._artifacts_dir,
                calibration_data_dir=self._calibration_data_dir,
                model_cfg=TIModelCfg(),
                precision_cfg=TIPrecisionCfg(tensor_bits=self._tensor_bits),
                calibration_cfg=calibration_cfg,
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            raise HTTPInternalServerError(reason=str(exc))

        print(f"Compilation time: {time.time() - start}")

        shutil.make_archive(
            base_name=str(self._working_dir / "artifacts"),
            format="zip",
            root_dir=str(self._artifacts_dir),
        )

        return self._working_dir / "artifacts.zip"

    def measure_latency(self, model: bytes) -> Dict[str, float]:
        """
        Method that allows to measure latency using Texas Instruments TDA4 with TIDL acceleration.

        Parameters
        ----------
        model : bytes
            ONNX model which latency we want to measure.
        Returns
        -------
        Dict[str, float]
            Dict with measured parameters

        """
        self._cleanup_working_dir()
        path_to_model_artifacts = self._compile_model(model, calibration_data=None)
        with path_to_model_artifacts.open(mode="rb") as model_artifacts_file:
            model_artifacts = model_artifacts_file.read()

        result = measure_latency_remote(
            host=self.ti_host,
            port=self.ti_port,
            model=model_artifacts,
        )

        return result


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ti-host",
        required=True,
        type=str,
        help="Hostname of IP address of latency measurement server on TI",
    )
    parser.add_argument(
        "--ti-port",
        required=True,
        type=int,
        help="Port of latency measurement server on TI",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host name or IP address of compilation server. Default value is '0.0.0.0'",
    )
    parser.add_argument("--port", type=int, default=15003, help="Port of compilation server. Default value is 15003")
    parser.add_argument("--working-dir", default="./working_dir", help="Path to the working directory for tmp files.")

    return parser.parse_args()


def main():
    args = parse()
    server = MainLatencyServer(
        host=args.host,
        port=args.port,
        ti_host=args.ti_host,
        ti_port=args.ti_port,
        working_dir=args.working_dir,
    )
    server.run()


if __name__ == "__main__":
    main()
