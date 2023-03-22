"""
Example usage of pose detector class
"""

import argparse

from alphapose.utils.config import update_config
from alphapose.utils.frame_detector import Pose

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cfg", type=str, required=True, help="experiment configure file name"
)
parser.add_argument(
    "--checkpoint", type=str, required=True, help="checkpoint file name"
)
parser.add_argument(
    "--min_box_area", type=int, default=0, help="min box area to filter out"
)
parser.add_argument(
    "--posebatch",
    type=int,
    default=64,
    help="pose estimation maximum batch size PER GPU",
)

args = parser.parse_args()
cfg = update_config(args.cfg)

if __name__ == "__main__":

    pose_model = Pose(
        cfg, args.checkpoint, args.posebatch, args.min_box_area, args.device
    )

    boxes, scores = [], []
    img = []

    result = pose_model.infer(img, boxes, scores)
