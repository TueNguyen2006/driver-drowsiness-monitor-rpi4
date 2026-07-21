from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from hybrid_system.pipeline import fourcc_to_str, open_camera_capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe camera frames and save raw snapshots.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fourcc", default="MJPG", help="Preferred camera FourCC, e.g. MJPG or YUYV")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--sleep-ms", type=float, default=100.0)
    parser.add_argument("--out-dir", default="camera_probe")
    return parser.parse_args()


def describe(frame: np.ndarray) -> str:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if frame.ndim == 3:
        channels = frame.reshape(-1, frame.shape[2]).mean(axis=0)
        channel_text = " bgr_mean=" + ",".join(f"{v:.2f}" for v in channels[:3])
    else:
        channel_text = ""
    return (
        f"shape={frame.shape} dtype={frame.dtype} "
        f"mean={float(gray.mean()):.2f} std={float(gray.std()):.2f} "
        f"min={int(gray.min())} max={int(gray.max())}{channel_text}"
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = open_camera_capture(
        args.camera_index,
        fourcc=args.fourcc,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(f"opened={cap.isOpened()} index={args.camera_index} requested_fourcc={args.fourcc}")
    if not cap.isOpened():
        return
    print(
        "props:",
        f"width={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}",
        f"height={cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}",
        f"fps={cap.get(cv2.CAP_PROP_FPS):.2f}",
        f"fourcc={fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))}",
    )

    saved = 0
    last_frame = None
    for idx in range(max(1, args.frames)):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"frame={idx} ret={ret} frame=None")
            time.sleep(args.sleep_ms / 1000.0)
            continue
        diff = 0.0
        if last_frame is not None and last_frame.shape == frame.shape:
            diff = float(np.mean(cv2.absdiff(last_frame, frame)))
        print(f"frame={idx} ret={ret} {describe(frame)} diff_prev={diff:.2f}")
        if saved < 5:
            path = out_dir / f"camera_{args.camera_index}_{idx:03d}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"saved={path}")
            saved += 1
        last_frame = frame.copy()
        time.sleep(args.sleep_ms / 1000.0)
    cap.release()


if __name__ == "__main__":
    main()
