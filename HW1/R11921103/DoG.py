import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        for n in range(self.num_octaves):
            oct = []
            for i in range(self.num_guassian_images_per_octave):
                if(n == 0 and i == 0):
                    oct.append(image)
                elif(n == 1 and i == 0):
                    w = gaussian_images[0][-1].shape[1] // 2
                    h = gaussian_images[0][-1].shape[0] // 2
                    resized = cv2.resize(gaussian_images[0][-1], (w, h), interpolation=cv2.INTER_NEAREST)
                    oct.append(resized)
                else:
                    oct.append(cv2.GaussianBlur(oct[0], (0, 0), sigmaX = self.sigma ** i))
            gaussian_images.append(oct)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for octave in gaussian_images:
            tmp = []
            for i in range(1, self.num_guassian_images_per_octave):
                tmp.append(cv2.subtract(octave[i], octave[i - 1]))
            dog_images.append(np.stack(np.array(tmp)))

        # for i in range(2):
        #     for j in range(4):
        #         img_normalized = cv2.normalize(dog_images[i][j], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #         cv2.imwrite('./plots/{}-{}.jpg'.format(i + 1, j + 1), img_normalized)
                
        # Step 3: Thresholding the value and Find local extremum (local maximum and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for n in range(self.num_octaves):
            if(n == 0):
                height, width = image.shape
            else:
                height, width = height // 2, width // 2

            for k in range(1, self.num_DoG_images_per_octave - 1):
                for i in range(1, height - 1):
                    for j in range(1, width - 1):
                        mid = dog_images[n][k, i, j]
                        values = np.delete(dog_images[n][k-1: k+2, i - 1:i + 2, j - 1:j + 2].flatten(), 13)
                        if(abs(mid) >= self.threshold):
                            if((mid >= np.max(values)) or (mid <= np.min(values))):
                                if(n == 0):
                                    keypoints.append([i, j])
                                else:
                                    keypoints.append([2 * i, 2* j])
        keypoints = np.array(keypoints)


        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
