import argparse
import pickle

import onnx
import requests


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--onnx-model", required=True, type=str, help="Path to ONNX model")
    parser.add_argument(
        "-c",
        "--calibration-data-zip",
        required=True,
        type=str,
        help="Path to the calibration data zip archive",
    )
    parser.add_argument(
        "-o",
        "--output-model",
        required=True,
        type=str,
        help="Path to the output zip",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host name or IP address of compilation server. Default value is '0.0.0.0'",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=15003,
        help="Port of compilation server. Default value is 15003",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=2 * 60 * 60,
        help="compilation time limit in seconds. Default value is 7200 (2 hours).",
    )
    return parser.parse_args()


def main():
    args = parse()

    onnx_model = onnx.load(args.onnx_model).SerializeToString()
    with open(args.calibration_data_zip, "rb") as calibration_data_zip_file:
        calibration_data = calibration_data_zip_file.read()

    print("Start compilation, please wait... (Compilation takes about 3-10 minutes)")
    response = requests.post(
        url=f"http://{args.host}:{args.port}/compile",
        data=pickle.dumps({"model": onnx_model, "calibration_data": calibration_data}),
        headers={"Content-Type": "application/octet-stream"},
        timeout=args.timeout,
    )

    if response.status_code == 200:
        with open(args.output_model, "wb") as output_model_file:
            output_model_file.write(response.content)

        print("Compiled model saved")
    else:
        raise RuntimeError(f"Expected status code is 200, got {response.status_code}; reason: {response.reason}")


if __name__ == "__main__":
    main()
