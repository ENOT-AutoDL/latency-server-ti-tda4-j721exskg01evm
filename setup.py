# pylint: disable=missing-module-docstring
from pathlib import Path

from setuptools import find_packages
from setuptools import setup

_SRC = Path("src")
_MODULE_NAME = "texas_instruments_latency_server"
_DIR_PATH = Path(__file__).parent.joinpath(_SRC).resolve()


setup(
    name="texas-instruments-latency-server",
    version="1.0.1",
    description="Latency measurement server/client for Texas Instruments board (j721exskg01evm)",
    author="ENOT LLC",
    author_email="enot@enot.ai",
    install_requires=[
        "enot-latency-server>=1.2.0",
        "numpy",
    ],
    extras_require={
        "device_server": [
            "onnxruntime-tidl==1.7.0",
        ],
        "main_server": [
            "dataclasses",
            "onnx==1.9.0",
            "onnxruntime-tidl==1.7.0",
            "protobuf==3.19.*",
        ],
        "minimal": [
            "onnx",
        ],
    },
    packages=find_packages(where=_DIR_PATH.as_posix()),
    package_dir={
        # workaround for develop mode
        # https://github.com/pypa/setuptools/issues/230
        "": _SRC.as_posix(),
        _MODULE_NAME: _SRC.joinpath(_MODULE_NAME).as_posix(),
    },
    entry_points={
        "console_scripts": [
            "ti-main-server = texas_instruments_latency_server.main_server:main",
            "ti-device-server = texas_instruments_latency_server.device_server:main",
            "ti-measure-latency = texas_instruments_latency_server.measure_latency:main",
            "ti-remote-compiler = texas_instruments_latency_server.compile_model:main",
            "ti-compiler = texas_instruments_latency_server.compiler.cli:compile_onnx_model",
        ],
    },
)
