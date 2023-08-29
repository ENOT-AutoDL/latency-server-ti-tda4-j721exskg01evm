import argparse
import tempfile
from pathlib import Path

from texas_instruments_latency_server.compiler.common import generate_fake_calibration_data
from texas_instruments_latency_server.compiler.compiler import TICompiler
from texas_instruments_latency_server.compiler.config import TIAccuracyLevel
from texas_instruments_latency_server.compiler.config import TICalibrationCfg
from texas_instruments_latency_server.compiler.config import TIDebugLevel
from texas_instruments_latency_server.compiler.config import TIModelCfg
from texas_instruments_latency_server.compiler.config import TIPrecisionCfg
from texas_instruments_latency_server.compiler.config import TITensorBits


def compile_onnx_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-onnx", type=str, required=True, help="Path to model onnx")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Path to output directory")
    parser.add_argument(
        "--calibration-data-dir",
        type=str,
        default="",
        help="Path to calibration data directory. If empty, false calibration data is used and the output model has an "
        "accuracy of 0 but is still useful for the measurement of latency. Empty by default.",
    )
    parser.add_argument(
        "--debug-level",
        type=int,
        default=TIDebugLevel.NO_DEBUG.value,
        choices=[e.value for e in TIDebugLevel],
        help="Compiler debug level",
    )
    parser.add_argument(
        "--tensor-bits",
        type=int,
        default=TITensorBits.TENSOR_8_BITS.value,
        choices=[e.value for e in TITensorBits],
        help="Tensors bitness. 8 by default.",
    )
    parser.add_argument(
        "--calibration-algorithm",
        default=TIAccuracyLevel.BASIC.name,
        choices=[e.name for e in TIAccuracyLevel],
        help="Calibration algorithm. Default value is 'BASIC'.",
    )
    args = parser.parse_args()

    if args.calibration_data_dir == "":
        parent_dir = Path(args.model_onnx).parent
        fake_calibration_dir = tempfile.TemporaryDirectory(dir=parent_dir)
        calibration_dir = Path(fake_calibration_dir.name)
        generate_fake_calibration_data(model_path=Path(args.model_onnx), output_dir=calibration_dir)
    else:
        fake_calibration_dir = None
        calibration_dir = Path(args.calibration_data_dir)

    debug_level = TIDebugLevel(args.debug_level)
    compiler = TICompiler(debug_level=debug_level)

    tensor_bits = TITensorBits(args.tensor_bits)
    accuracy_level = TIAccuracyLevel[args.calibration_algorithm]
    compiler.compile(
        model_path=args.model_onnx,
        output_dir=args.output_dir,
        calibration_data_dir=calibration_dir,
        model_cfg=TIModelCfg(),
        precision_cfg=TIPrecisionCfg(tensor_bits=tensor_bits),
        calibration_cfg=TICalibrationCfg(
            accuracy_level=accuracy_level,
            calibration_iterations=1 if accuracy_level is TIAccuracyLevel.BASIC else 5,
        ),
    )

    # this "if" allow to avoid unexpected cleanup of tmp directory during compilation, because of GC
    if fake_calibration_dir is not None:
        fake_calibration_dir.cleanup()


if __name__ == "__main__":
    compile_onnx_model()
