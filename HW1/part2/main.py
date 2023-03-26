import numpy as np
import cv2
import argparse
import os
import csv
from JBF import Joint_bilateral_filter
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    with open(args.setting_path, 'r') as f:
        settings = list(csv.reader(f))
    gs_settings = [np.array(x).astype(np.float64) for x in settings[1:-1]]
    gs_images = [np.sum(img_rgb * s, axis=2) for s in gs_settings]
    gs_images.append(img_gray)
    sigma_s, sigma_r = int(settings[-1][1]), float(settings[-1][3])
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    jbf_out = [JBF.joint_bilateral_filter(img_rgb, guidance).astype(np.uint8) for guidance in gs_images]
    jbf_error = [np.sum(np.abs(jbf.astype('int32')-bf_out.astype('int32'))) for jbf in jbf_out]
    
    for i in range(5):
        print(gs_settings[i], jbf_error[i])
    print('cv2.COLOR_BGR2GRAY', jbf_error[5])

    for i, (color, gs) in enumerate(zip(jbf_out, gs_images[:-1])):
        cv2.imwrite('./plots/1_{}_gray.jpg'.format(i), gs)
        cv2.imwrite('./plots/1_{}_color.jpg'.format(i), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./plots/{}.jpg'.format('1_cv2_gray'), gs_images[-1])
    cv2.imwrite('./plots/{}.jpg'.format('1_cv2_color'), cv2.cvtColor(jbf_out[-1], cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main()