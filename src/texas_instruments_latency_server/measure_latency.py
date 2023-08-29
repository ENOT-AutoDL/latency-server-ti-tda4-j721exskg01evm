import argparse

import onnx
from enot_latency_server.client import measure_latency_remote


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--onnx-model", required=True, type=str, help="Path to ONNX model")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host name or IP address of server. If you want to measure latency on NPU, pass compilation server host. "
        "If you want to measure latency on CPU, pass CPU latency server host. Default value is '0.0.0.0'",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=15003,
        help="Port of server. If you want to measure latency on NPU, pass compilation server port. "
        "If you want to measure latency on CPU, pass CPU latency server port. Default value is 15003",
    )
    return parser.parse_args()


def main():
    args = parse()
    onnx_model = onnx.load(args.onnx_model)
    print("Start latency measurement, please wait... (Average measurement takes about 5 minutes)")
    result = measure_latency_remote(
        model=onnx_model.SerializeToString(),
        host=args.host,
        port=args.port,
    )
    print(result)


if __name__ == "__main__":
    main()
