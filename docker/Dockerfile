FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.6 python3.6-distutils python3-pip graphviz

RUN mkdir /opt/latency_server_project/
COPY ./ /opt/latency_server_project

RUN mkdir /opt/latency_server/
RUN mkdir /opt/latency_server/log
RUN mkdir /opt/latency_server/working_dir

RUN python3.6 -m pip install pip -U
RUN python3.6 -m pip install /opt/latency_server_project/3rd_party/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
RUN python3.6 -m pip install /opt/latency_server_project/[main_server]
RUN tar -xvf /opt/latency_server_project/3rd_party/tidl_tools.tar.gz -C /opt/latency_server/

RUN rm -rf /opt/latency_server_project/

ENV TIDL_TOOLS_PATH=/opt/latency_server/tidl_tools
ENV LD_LIBRARY_PATH=/opt/latency_server/tidl_tools

CMD ti-main-server --host '0.0.0.0' --port 15003 --ti-host "$TI_HOST" --ti-port "$TI_PORT" --working-dir /opt/latency_server/working_dir
