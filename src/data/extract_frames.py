import argparse
import os
from pathlib import Path

import cv2


def extract_frames(video_path: str, output_dir: str, interval: float) -> None:
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Frame extractor")
    print("=" * 60)
    print(f"Video    : {video_path}")
    print(f"Output   : {output_dir}")
    print(f"Interval : {interval}s")
    print("=" * 60)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps * interval)))

    existing = [
        int(f.stem) for f in output_dir.glob("*.jpg") if f.stem.isdigit()
    ]
    saved_count = max(existing, default=0)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            saved_count += 1
            filename = output_dir / f"{saved_count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"[{saved_count:4d}] saved {filename} (frame={frame_index})")

        frame_index += 1

    cap.release()

    print()
    print("=" * 60)
    print("Frame extraction completed.")
    print(f"Images saved: {saved_count}")
    print(f"Directory   : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from a video at a fixed time interval."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save extracted images."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between extracted frames."
    )
    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        interval=args.interval,
    )
