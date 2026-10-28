import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_images(titles, images, cmap='gray'):
    n = len(images)
    plt.figure(figsize=(4 * n, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 读取灰度图像
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)



def linear_transform(image, a=1.0, b=0):
    img = image.astype(np.float32)
    img = a * img + b
    return np.clip(img, 0, 255).astype(np.uint8)

linear = linear_transform(image, a=1.2, b=30)
show_images(['Original', 'Linear Transformed'], [image, linear])


def log_transform(image):
    img = image.astype(np.float32)
    c = 255 / np.log(1 + np.max(img))
    log_img = c * np.log(1 + img)
    return np.clip(log_img, 0, 255).astype(np.uint8)

log_img = log_transform(image)
show_images(['Original', 'Log Transformed'], [image, log_img])



def inverse_log_transform(image):
    img = image.astype(np.float32)
    img = img / 255.0
    exp_img = np.exp(img) - 1
    exp_img = exp_img / np.max(exp_img) * 255
    return np.clip(exp_img, 0, 255).astype(np.uint8)

inv_log = inverse_log_transform(image)
show_images(['Original', 'Inverse Log'], [image, inv_log])


def gamma_transform(image, gamma=1.0):
    img = image.astype(np.float32) / 255.0
    gamma_img = np.power(img, gamma)
    return np.clip(gamma_img * 255, 0, 255).astype(np.uint8)

gamma_img = gamma_transform(image, gamma=0.5)
show_images(['Original', 'Gamma Corrected'], [image, gamma_img])



def invert_transform(image):
    return 255 - image

inverted = invert_transform(image)
show_images(['Original', 'Inverted'], [image, inverted])



def piecewise_linear(image, r1, s1, r2, s2):
    def interpolate(r):
        if r < r1:
            return (s1 / r1) * r
        elif r < r2:
            return ((s2 - s1) / (r2 - r1)) * (r - r1) + s1
        else:
            return ((255 - s2) / (255 - r2)) * (r - r2) + s2

    vectorized = np.vectorize(interpolate)
    return vectorized(image).astype(np.uint8)

pl_img = piecewise_linear(image, r1=70, s1=30, r2=180, s2=220)
show_images(['Original', 'Piecewise Linear'], [image, pl_img])


def threshold_transform(image, thresh=128):
    _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return binary

binary = threshold_transform(image, thresh=100)
show_images(['Original', 'Binary'], [image, binary])


