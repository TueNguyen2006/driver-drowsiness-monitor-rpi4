import os
import sys
import time

import config as cfg
import state

LOCAL_SITE_PACKAGES = os.path.join(os.path.dirname(__file__), ".vendor")
if os.path.isdir(LOCAL_SITE_PACKAGES) and LOCAL_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, LOCAL_SITE_PACKAGES)

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


def resolve_arduino_port():
    if cfg.ARDUINO_PORT:
        return cfg.ARDUINO_PORT

    if list_ports is None:
        return None

    ports = list(list_ports.comports())
    for port in ports:
        description = " ".join(
            filter(None, [port.device, port.description, getattr(port, "manufacturer", None)])
        )
        normalized = description.lower()
        if "arduino" in normalized or "ch340" in normalized or "usb serial" in normalized:
            return port.device

    if len(ports) == 1:
        return ports[0].device

    return None


def handle_button_command(command):
    if command == "RESTART_INFERENCE":
        if state.mode in {"infer", "infer_paused"}:
            state.request_recalibrate = True
            print("[ARDUINO] Inference restart requested.")
        return

    if command == "TOGGLE_INFER_PAUSED":
        if state.mode in {"infer", "infer_paused"}:
            state.running_inference = not state.running_inference
            print(f"[ARDUINO] Inference paused = {not state.running_inference}")


def start_arduino_listener():
    if serial is None:
        print("[ARDUINO] pyserial is not installed. Arduino button control is disabled.")
        return

    while True:
        port = resolve_arduino_port()
        if not port:
            time.sleep(cfg.ARDUINO_RETRY_DELAY)
            continue

        try:
            with serial.Serial(port, cfg.ARDUINO_BAUDRATE, timeout=1) as serial_conn:
                print(f"[ARDUINO] Listening on {port} at {cfg.ARDUINO_BAUDRATE} baud.")
                time.sleep(2)

                while True:
                    raw_command = serial_conn.readline().decode(errors="ignore").strip()
                    if raw_command:
                        handle_button_command(raw_command)

        except Exception as exc:
            print(f"[ARDUINO] Connection error: {exc}")
            time.sleep(cfg.ARDUINO_RETRY_DELAY)
