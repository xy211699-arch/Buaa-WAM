#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio


def main():
    p = argparse.ArgumentParser(description="Build preview.mp4 from a collected air-hockey episode")
    p.add_argument("episode_dir", type=Path)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()

    episode = a.episode_dir.expanduser().resolve()
    rgb = episode / "rgb"
    if not rgb.is_dir():
        raise FileNotFoundError(rgb)

    frames = sorted(x for x in rgb.iterdir() if x.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not frames:
        raise RuntimeError(f"no frames under {rgb}")

    fps = a.fps
    meta = episode / "meta.json"
    if fps is None and meta.is_file():
        try:
            fps = float(json.loads(meta.read_text(encoding="utf-8")).get("fps", 30.0))
        except Exception:
            fps = 30.0
    fps = 30.0 if fps is None else fps
    if fps <= 0:
        raise ValueError("fps must be > 0")

    out = a.output.expanduser().resolve() if a.output else episode / "preview.mp4"
    with imageio.get_writer(str(out), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None) as w:
        for f in frames:
            w.append_data(imageio.imread(f))

    print("episode:", episode)
    print("frames :", len(frames))
    print("fps    :", fps)
    print("output :", out)


if __name__ == "__main__":
    main()
