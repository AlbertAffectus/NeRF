# get my data here:
# https://drive.google.com/drive/folders/1s0_kPLkJ5ueCYbr4-Ddqp5VcRXWdqnbv?usp=sharing

import os
import argparse
import torch
import numpy as np
import json
import cv2
from concurrent.futures import ThreadPoolExecutor
import pickle


def init_rays(h, w, focal, pose):
    K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]])

    i, j = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2])/K[0][0], -(j - K[1][2])/K[1][1], -np.ones_like(i)], -1)
    
    # rotate camera coordinate to world coordinate
    dirs = np.sum(dirs[..., np.newaxis, :] * pose[:3, :3], -1)
    # translate camera coordinate to world coordinate
    origins = np.broadcast_to(pose[:3,-1], np.shape(dirs))
    
    return origins, dirs


def ready_data(data_path, parent_path="", num_threads=8, h=800, w=800):
    # open the json file
    with open(data_path) as f:
        meta_data = json.load(f)
        cam_angle_x = meta_data['camera_angle_x']

        # calculate the focal length
        focal = 0.5 * 800 / np.tan(0.5 * cam_angle_x)
        focal *= w / 800

        print("focal: ", focal)

        num_frames = len(meta_data['frames'])
        data = np.empty((num_frames, h * w, 9), dtype=float)

        def read_and_resize(file_path):
            rgba = cv2.imread(os.path.join(
                parent_path, file_path), cv2.IMREAD_UNCHANGED)

            rgba[rgba[:, :, 3] == 0] = [255, 255, 255, 255]
            rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)[..., :3]
            rgb = cv2.resize(rgb, (w, h))
            rgb = rgb.astype(np.float32) / 255.
            return rgb

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            frame_paths = [frame['file_path'] +
                           ".png" for frame in meta_data['frames']]
            rgbs = list(executor.map(read_and_resize, frame_paths))

        for i, frame in enumerate(meta_data['frames']):
            print("Processing frame: ", i)
            rgb = rgbs[i]
            # pose is the first 3 rows and first 4 columns of the transform matrix
            pose = np.array(frame['transform_matrix'])[:3, :4]

            # get the rays
            origins, dirs = init_rays(h, w, focal, pose)

            print("origins.shape: ", origins.shape)
            print("dirs.shape: ", dirs.shape)
            print("rgb.shape: ", rgb.shape)

            # flatten each of origins, dirs, rgb
            origins = origins.reshape(-1, 3)
            dirs = dirs.reshape(-1, 3)
            rgb = rgb.reshape(-1, 3)

            d = np.concatenate([origins, dirs, rgb], axis=1)

            data[i] = d

        data = data.reshape(-1, 9)
        print("data.shape: ", data.shape)
        return data, focal


def main(args):
    w = args.w
    outfile = args.outfile
    parent_path = args.parent_path
    
    training_rays, focal = ready_data(parent_path + "/meta.json", parent_path=parent_path, num_threads=8, h=w, w=w)

    data = {
        'rays': training_rays,
        'h': w,
        'w': w,
        'focal': focal
    }

    with open(os.path.join(parent_path, outfile), "wb") as f:
        pickle.dump(data, f)
        print("Saved training rays to: ", os.path.join(parent_path, outfile))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w', type=int, default=100, help='width of the image')
    parser.add_argument('--outfile', type=str, help='filename for the output pickle file')
    parser.add_argument('--parent_path', type=str, help='path to parent directory of image directory and metadata file')

    args = parser.parse_args()

    main(args)

# python3 preprocess.py --w 100 --outfile mic_100.pkl --parent_path data/imgs/mic
# python3 preprocess.py --w 100 --outfile dog_100.pkl --parent_path data/imgs/hotdog

