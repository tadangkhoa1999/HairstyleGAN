import numpy as np
import cv2
import argparse
import os
import math

def read_transparent_png(filename):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)


def get_uper_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d

        if y3 < y4:
            return np.array([x3, y3])
        else:
            return np.array([x4, y4])


def mergeHair(face, hair):
    rows, cols, ch = face.shape
    for i in range(rows):
        for j in range(cols):
            if (hair[i, j, :] != [0, 0, 0]).all() and (hair[i, j, :] != [255, 255, 255]).all():
                face[i, j, :] = hair[i, j, :]

    return face


def affineTransformHair(face_path, hair_path, face_landmark_path, hair_landmark_path):
    # load pts1
    pts1 = np.float32(np.load(hair_landmark_path))
    pts1 = pts1[pts1[:, 0].argsort()]

    # Load landmark
    lm = np.array(np.load(face_landmark_path))
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nostrils      = lm[31 : 36]  # top-down
    nostrils_avg = np.mean(lm_nostrils, axis=0)
    eyebrow_avg = (np.mean(lm_eyebrow_left, axis=0) + np.mean(lm_eyebrow_right, axis=0)) * 0.5

    # Left ear, Right ear, Forehead avg -> pts2
    left_ear = lm[0]
    right_ear = lm[16]
    forehead_avg = 2 * eyebrow_avg - nostrils_avg
    pts2 = np.float32([left_ear, forehead_avg, right_ear])

    # Load face
    face = cv2.imread(face_path, cv2.IMREAD_COLOR)
    rows, cols, ch = face.shape

    # Load hair -> resize
    hair = read_transparent_png(hair_path)
    newHair = np.zeros(face.shape, np.uint8)
    newHair[0:hair.shape[0], 0:hair.shape[1]] = hair

    # Affine Transform Hair and merge hair with face
    resize_score = np.linalg.norm(pts2[0] - pts2[2]) / np.linalg.norm(pts1[0] - pts1[2])
    r0 = resize_score * np.linalg.norm(pts1[0] - pts1[1])
    r1 = resize_score * np.linalg.norm(pts1[1] - pts1[2])
    pts2[1] = get_intersections(pts2[0, 0], pts2[0, 1], r0, pts2[2, 0], pts2[2, 1], r1)
    M = cv2.getAffineTransform(pts1, pts2)
    newHair = cv2.warpAffine(newHair, M, (cols, rows))
    
    return mergeHair(face, newHair)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge hair-faces from input images, hairstyle', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--face_dir', default='aligned_images', help='Directory for storing face images')
    parser.add_argument('--hair_dir', default='hairs', help='Directory for storing hair images')
    parser.add_argument('--merge_hair_dir', default='merge_hair_images', help='Directory for storing merge hair images')
    parser.add_argument('--face_landmark_dir', default='landmarks', help='Directory for storing face landmarks')
    parser.add_argument('--hair_landmark_dir', default='hair_landmark', help='Directory for storing hair landmarks')

    args, other_args = parser.parse_known_args()

    for img_name in os.listdir(args.face_dir):
        img_path = os.path.join(args.face_dir, img_name)
        img_number = img_name.split('_')[0]
        face_landmark_path = os.path.join(args.face_landmark_dir, img_number + '.npy')

        for hair_name in os.listdir(args.hair_dir):
            hair_path = os.path.join(args.hair_dir, hair_name)
            hair_numer = hair_name.split('.')[0]
            hair_landmark_path = os.path.join(args.hair_landmark_dir, hair_numer + '.npy')

            merge_image_name = os.path.join(args.merge_hair_dir, str(img_number) + '_' + str(hair_numer) + '.png')
            merge_image = affineTransformHair(img_path, hair_path, face_landmark_path, hair_landmark_path)

            cv2.imwrite(merge_image_name, merge_image)