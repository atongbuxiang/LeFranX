import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="frame_000075.png")
    parser.add_argument("-o", "--out", default="depth_15_35cm_vis.png")
    args = parser.parse_args()

    depth = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise SystemExit(f"Could not read image: {args.image}")

    valid = depth > 0
    if valid.any():
        print(
            depth.dtype,
            int(depth.min()),
            int(depth.max()),
            np.percentile(depth[valid], [1, 50, 99]),
        )
    else:
        print(depth.dtype, "no positive depth pixels")

    # 只显示 15cm-35cm（假设 depth 单位为毫米，与 RealSense 常见设置一致）
    vis = np.clip((depth.astype(np.float32) - 150) / (350 - 150), 0, 1)
    vis = (vis * 255).astype(np.uint8)

    out_path = Path(args.out)
    cv2.imwrite(str(out_path), vis)
    print(f"Saved visualization: {out_path.resolve()}")

    try:
        cv2.imshow("depth 15-35cm", vis)
        cv2.waitKey(0)
    except cv2.error as e:
        print(
            "OpenCV GUI not available (e.g. opencv-python-headless). "
            "Install GUI build: pip install 'opencv-python>=4' "
            "after removing opencv-python-headless if needed.\n"
            f"Original error: {e}"
        )


if __name__ == "__main__":
    main()
