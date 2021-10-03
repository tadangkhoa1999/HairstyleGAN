import os
import sys
sys.path.append(os.getcwd())
import bz2
import argparse

from tqdm import tqdm
import numpy as np

import utils
from modules.landmarks.face_alignment import image_align
from modules.landmarks.landmarks_detector import LandmarksDetector


LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    print(src_path)
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib
    python align_images.py /raw_images /aligned_images
    """
    parser = argparse.ArgumentParser(description='Align faces from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('raw_dir', help='Directory with raw images for face alignment')
    parser.add_argument('aligned_dir', help='Directory for storing aligned images')
    parser.add_argument('--landmarks_path', help='Directory for storing landmarks', required=True)
    parser.add_argument('--aligned_landmarks_path', help='Directory for storing landmarks', required=True)
    parser.add_argument('--output_size', default=1024, help='The dimension of images for input to the model', type=int)
    parser.add_argument('--x_scale', default=1, help='Scaling factor for x dimension', type=float)
    parser.add_argument('--y_scale', default=1, help='Scaling factor for y dimension', type=float)
    parser.add_argument('--em_scale', default=0.1, help='Scaling factor for eye-mouth distance', type=float)
    parser.add_argument('--use_alpha', default=False, help='Add an alpha channel for masking', type=bool)

    args, other_args = parser.parse_known_args()

    landmarks_model_path = unpack_bz2(utils.open_url(LANDMARKS_MODEL_URL, return_filename=True))
    RAW_IMAGES_DIR = args.raw_dir
    ALIGNED_IMAGES_DIR = args.aligned_dir
    LANDMARKS_DIR=args.landmarks_path
    ALIGNED_LANDMARKS_DIR=args.aligned_landmarks_path

    print('Aligning images ...')
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in tqdm(os.listdir(RAW_IMAGES_DIR)):
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        fn = face_img_name = '%s.png' % (os.path.splitext(img_name)[0])
        if os.path.isfile(fn):
            continue
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            np.save(os.path.join(LANDMARKS_DIR, os.path.splitext(img_name)[0] + '.npy'), face_landmarks)
            aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
            image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=args.output_size, x_scale=args.x_scale, y_scale=args.y_scale, em_scale=args.em_scale, alpha=args.use_alpha)
            for i, new_face_landmarks in enumerate(landmarks_detector.get_landmarks(aligned_face_path), start=1):
                np.save(os.path.join(ALIGNED_LANDMARKS_DIR, os.path.splitext(img_name)[0] + '.npy'), new_face_landmarks)
