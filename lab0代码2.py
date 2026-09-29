"""OpenCV aloe stereo pair: disparity and relative depth estimation.

Run: python stereo_depth_estimation.py DATA_DIR OUTPUT_DIR
DATA_DIR contains aloeL.jpg, aloeR.jpg and optionally aloeGT.png from
OpenCV's samples/data directory. Requires opencv-python and numpy.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    left = cv2.imread(str(data_dir / 'aloeL.jpg'))
    right = cv2.imread(str(data_dir / 'aloeR.jpg'))
    if left is None or right is None or left.shape != right.shape:
        raise ValueError('Missing or mismatched aloeL.jpg and aloeR.jpg')
    left = cv2.pyrDown(left)
    right = cv2.pyrDown(right)
    block = 5
    channels = 3
    min_disp = 16
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=96, blockSize=block,
        P1=8 * channels * block**2, P2=32 * channels * block**2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = matcher.compute(left, right).astype(np.float32) / 16.0
    valid = np.isfinite(disparity) & (disparity > min_disp)
    # With no physical focal length and baseline for this pair, 1/d is a
    # relative depth proxy. Its units are inverse pixels, not metres.
    relative_depth = np.full(disparity.shape, np.nan, np.float32)
    relative_depth[valid] = 1.0 / disparity[valid]
    colored_disp = cv2.applyColorMap(np.clip((disparity - 16) * 255 / 96, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored_disp[~valid] = (0, 0, 0)
    lo, hi = np.nanpercentile(relative_depth, [2, 98])
    depth_u8 = np.clip((relative_depth - lo) * 255 / (hi - lo), 0, 255)
    depth_u8 = np.nan_to_num(depth_u8).astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    colored_depth[~valid] = (0, 0, 0)
    cv2.imwrite(str(output_dir / 'left.png'), left)
    cv2.imwrite(str(output_dir / 'right.png'), right)
    cv2.imwrite(str(output_dir / 'disparity.png'), colored_disp)
    cv2.imwrite(str(output_dir / 'relative_depth.png'), colored_depth)
    np.save(output_dir / 'disparity_px.npy', disparity)
    metrics = {
        'opencv_version': cv2.__version__, 'image_size': [left.shape[1], left.shape[0]],
        'min_disparity_px': min_disp, 'num_disparities': 96, 'block_size': block,
        'valid_pixel_fraction': float(valid.mean()),
        'median_disparity_px': float(np.median(disparity[valid])),
        'relative_depth_definition': '1 / disparity_px, inverse-pixel units',
    }
    gt_path = data_dir / 'aloeGT.png'
    if gt_path.exists():
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 2.0
        compared = valid & (gt > 0)
        err = np.abs(disparity[compared] - gt[compared])
        metrics.update({
            'gt_coverage_fraction': float(compared.mean()),
            'mae_disparity_px': float(err.mean()),
            'rmse_disparity_px': float(np.sqrt(np.mean(err**2))),
            'bad3_fraction': float(np.mean(err > 3.0)),
            'compared_pixels': int(compared.sum()),
        })
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python stereo_depth_estimation.py DATA_DIR OUTPUT_DIR')
    main(Path(sys.argv[1]), Path(sys.argv[2]))