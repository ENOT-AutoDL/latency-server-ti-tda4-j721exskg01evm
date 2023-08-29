#!/bin/bash

set -e

PROJECT_DIR=$(dirname "$0")
LATENCY_SERVER_HOME_DIR=/opt/latency_server
LATENCY_SERVER_LOG_DIR=$LATENCY_SERVER_HOME_DIR/log
LATENCY_SERVER_WORKING_DIR=$LATENCY_SERVER_HOME_DIR/working_dir


uninstall_service() {
  local service_name=$1
  local service_unit=$service_name.service

  # uninstall service
  if systemctl list-units --full -all | grep -Fq "$service_unit"; then
    echo "removing $service_name from systemd"

    systemctl stop "$service_unit"
    systemctl disable "$service_unit"
    rm -f /etc/systemd/system/"$service_unit"
    systemctl daemon-reload
    systemctl reset-failed

    echo "$service_name is removed from systemd"
  fi
}

uninstall() {
  # uninstall supervisor service
  uninstall_service "supervisor"
  # remove supervisor configs
  if [ -d /etc/supervisor ]; then
    rm -rf /etc/supervisor
    echo "supervisor configuration files is removed"
  fi

  # remove latency server home dir
  if [ -d $LATENCY_SERVER_HOME_DIR ]; then
    rm -rf $LATENCY_SERVER_HOME_DIR
    echo "log/configuration/tmp latency server files is removed"
  fi

  # uninstall latency server python package
  python3 -m pip uninstall -y texas-instruments-latency-server
  echo "latency server python package is removed"
}

install() {
  # prepare dirs
  mkdir -p $LATENCY_SERVER_HOME_DIR
  mkdir -p $LATENCY_SERVER_LOG_DIR
  mkdir -p $LATENCY_SERVER_WORKING_DIR
  echo "latency server directories is created"
  # copy tidl tools
  tar -xvf "$PROJECT_DIR"/3rd_party/tidl_tools.tar.gz -C $LATENCY_SERVER_HOME_DIR
  echo "tidl runtime is copied"

  # update pip
  python3 -m pip install pip -U
  # install latency server package
  python3 -m pip install "$PROJECT_DIR"'[device_server]'
  echo "latency server python package is installed"

  # install supervisor package
  python3 -m pip install pip supervisor
  echo "supervisor python package is installed"
  # install supervisor configs
  mkdir -p /etc/supervisor/
  SUPERVISOR_DIR="$PROJECT_DIR"/3rd_party/supervisor/device-server
  cp "$SUPERVISOR_DIR"/etc/supervisor/supervisord.conf /etc/supervisor/supervisord.conf
  cp "$SUPERVISOR_DIR"/etc/systemd/system/supervisor.service /etc/systemd/system/supervisor.service
  echo "supervisor configs is copied"
  # install supervisor service
  systemctl enable supervisor.service
  systemctl start supervisor.service
  echo "supervisor service is installed"
}

check_installation(){
  echo ""
  echo "===================================="
  echo "checking latency server installation"
  echo "===================================="
  echo ""

  python_package_status=$(python3 -c "import texas_instruments_latency_server" &> /dev/null && echo -n "OK" || echo -n "ERROR")
  echo -e "latency server python package\t\t$python_package_status"
  home_dir_status=$([ -d $LATENCY_SERVER_HOME_DIR ] && echo -n "OK" || echo -n "ERROR")
  echo -e "latency server home directory\t\t$home_dir_status"

  service_installed_status=$(systemctl list-units --full --all | grep -Fq supervisor.service && echo -n "OK" || echo -n "ERROR")
  echo -e "supervisord service is installed\t$service_installed_status"
  service_is_active_status=$(systemctl is-active --quiet supervisor.service && echo -n "OK" || echo -n "ERROR")
  echo -e "supervisord service is active\t\t$service_is_active_status"
}

safe_install(){
  uninstall
  install
  sleep 3
  check_installation
}

update(){
  safe_install
}

safe_install
