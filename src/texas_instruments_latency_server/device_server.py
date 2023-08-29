import argparse
import asyncio
import io
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict
from typing import Union

import numpy as np
from aiohttp.web_runner import GracefulExit
from enot_latency_server.server import LatencyServer

from texas_instruments_latency_server.ti_ort_model import TIOnnxruntimeModel


class DeviceLatencyServer(LatencyServer):
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 15003,
        warmup: int = 50,
        repeat: int = 50,
        number: int = 50,
        working_dir: Union[str, Path] = "./working_dir",
        reboot_after_measure: bool = False,
    ):
        """
        Server ctor.

        Parameters
        ----------
        host : str
             Host name or IP address. Default value is '0.0.0.0'.
        port : int
             Port. Default value is 15003.
        warmup : int
            Run 'warmup' warmup iterations before measuring the performance.
            Default is 50.
        repeat : int
            Run 'repeat' inference iterations
            Default is 50.
        number : int
            Number of iterations in each 'repeat' iteration
            Default is 50.
        working_dir : Union[str, Path]
            Path to the working directory for tmp files.
        reboot_after_measure : bool
            If True, the board will be rebooted after measurement is performed. Default value is False.

        """
        super().__init__(host=host, port=port)

        self.warmup = warmup
        self.repeat = repeat
        self.number = number
        self.working_dir = Path(working_dir).resolve()

        self._reboot_after_measure = reboot_after_measure
        self._reboot_on_exit = False

    def run(self):
        super().run()
        if self._reboot_on_exit:
            subprocess.run("reboot", shell=True)

    def create_ti_model(self, model: bytes) -> TIOnnxruntimeModel:
        """
        Creates TIOnnxruntimeModel for latency measurement on TDA4.

        Parameters
        ----------
        model : bytes
            Model artifacts compressed by zipfile or onnx model.

        Returns
        -------
        TIOnnxruntimeModel
            Model for latency measurement.

        """
        # cleanup working dir
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)

        self.working_dir.mkdir(parents=True)

        # Extract data
        artifacts_zip = io.BytesIO(model)
        try:
            if zipfile.is_zipfile(artifacts_zip):
                with zipfile.ZipFile(artifacts_zip, mode="r") as zf:
                    zf.extractall(self.working_dir)
                model_path = self.working_dir
            else:
                model_path = self.working_dir / "model.onnx"
                with model_path.open(mode="wb") as model_file:
                    model_file.write(model)
        finally:
            artifacts_zip.close()

        return TIOnnxruntimeModel(model_path=model_path)

    def compute_model_latency(self, ti_model: TIOnnxruntimeModel) -> float:
        """
        Computes model latency.

        Parameters
        ----------
        ti_model : TIOnnxruntimeModel
            Model which latency we want to compute.

        Returns
        -------
        float
            Model latency.

        """
        for _ in range(self.warmup):
            ti_model.benchmark_run()

        times = [ti_model.benchmark_run() for _ in range(self.repeat * self.number)]
        return float(np.mean(times)) / ti_model.batch_size

    def measure_latency(self, model: bytes) -> Dict[str, float]:
        """
        Method that allows to measure latency using Texas Instruments TDA4.

        Parameters
        ----------
        model : bytes
            ONNX model or zip archive with model artifacts which latency we want to measure.

        Returns
        -------
        Dict[str, float]
            Dict with measured parameters

        """
        ti_model = self.create_ti_model(model)
        latency = self.compute_model_latency(ti_model)

        stats = ti_model.collect_ti_stats()
        stats["latency"] = latency
        stats["ORT_overhead_ms"] = latency - stats["total_ms"]

        if self._reboot_after_measure:
            self._reboot_on_exit = True
            self._shutdown_server(delay=3)

        return stats

    def _shutdown_server(self, delay: Union[int, float]) -> None:
        async def graceful_exit():
            await asyncio.sleep(delay)
            raise GracefulExit()

        asyncio.run_coroutine_threadsafe(graceful_exit(), self._loop)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host name or IP address. Default value is '0.0.0.0'",
    )
    parser.add_argument("--port", type=int, default=15003, help="Server port. Default value is 15003")
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Run 'warmup' warmup iterations before measuring the performance",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Run 'repeat' inference iterations for latency measurement",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=50,
        help="Number of iterations in each 'repeat' iteration for latency measurement",
    )
    parser.add_argument("--working-dir", default="./working_dir", help="Path to the working directory for tmp files.")
    parser.add_argument(
        "--reboot-after-measure",
        action="store_true",
        help="the board will be rebooted after measurement is performed.",
    )

    return parser.parse_args()


def main():
    args = parse()
    server = DeviceLatencyServer(
        host=args.host,
        port=args.port,
        warmup=args.warmup,
        repeat=args.repeat,
        number=args.number,
        working_dir=args.working_dir,
        reboot_after_measure=args.reboot_after_measure,
    )
    server.run()


if __name__ == "__main__":
    main()
