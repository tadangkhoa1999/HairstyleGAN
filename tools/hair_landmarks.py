import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import cv2


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


def d(a, b):
    return np.linalg.norm(a-b)


def draw_hair_landmarks(face_dir, face_landmark_dir):
    for img_name in os.listdir(face_dir):
        img_path = os.path.join(face_dir, img_name)
        img_number = img_name.split('.')[0]
        face_landmark_path = os.path.join(face_landmark_dir, img_number + '.npy')

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

        face = cv2.imread(img_path, cv2.IMREAD_COLOR)
        pts2 = np.array([left_ear, forehead_avg, right_ear])
        for i in pts2:
            face = cv2.circle(face, (int(i[0]), int(i[1])), radius=2, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(img_name, face)


def get_hair_landmarks(hair_dir):
    for hair_image in os.listdir(hair_dir):
        hair_path = os.path.join(hair_dir, hair_image)
        hair_number = hair_image.split('.')[0]

        hair = read_transparent_png(hair_path)
        rows, cols, ch = hair.shape

        hair_landmark = []
        for i in range(rows):
            for j in range(cols):
                if not (hair[i, j, :] == [0, 0, 255]).all():
                    continue

                for x in hair_landmark:
                    if d(np.array([j, i]), x) < 10:
                        break
                else:
                    hair_landmark.append(np.array([j, i]))

        hair_landmark = np.array(hair_landmark)
        np.save(f'{hair_number}.npy', hair_landmark)


if __name__ == "__main__":
    face_dir = 'data/raw_images'
    face_landmark_dir = 'exp/features/landmarks'
    hair_dir = 'data/hairs'

    get_hair_landmarks(hair_dir)
    # draw_hair_landmarks(face_dir, face_landmark_dir)
