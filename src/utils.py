import cv2
import os
import numpy as np


def jpeg_to_video(path):
    os.system(
        "ffmpeg -r 30 -i {}/img%01d.jpeg -vcodec mpeg4 -y extracted_wms.mp4".
        format(path))


class ArnoldTransform:

    def __init__(self, img):
        self.img = img
        self.count = 0
        self.len = self.img.shape[0]
        x, y = np.meshgrid(range(self.len), range(self.len))
        self.x_map = (x + y) % self.len
        self.y_map = (2 * x + y) % self.len

    def transform(self, steps=1):
        for i in range(steps):
            self.img = self.img[self.x_map, self.y_map]
            self.count += 1
        return self.img


def randomize_channel(channel, blk_shape=(8, 8)):
    rows = channel.shape[0] // blk_shape[0] * blk_shape[0]
    cols = channel.shape[1] // blk_shape[1] * blk_shape[1]
    blks = np.array([[
        channel[i:i + blk_shape[0], j:j + blk_shape[1]]
        for j in range(0, cols, blk_shape[1])
    ] for i in range(0, rows, blk_shape[0])])
    shape = blks.shape
    blks = blks.reshape(-1, blk_shape[0], blk_shape[1])
    np.random.RandomState(0).shuffle(blks)
    full_res = np.copy(channel)
    res = np.concatenate(np.concatenate(blks.reshape(shape), 1), 1)
    full_res[:rows, :cols] = res
    return full_res


def randomize_img(img):
    for i in range(3):
        img[:, :, i] = randomize_channel(img[:, :, i])
    return img


def derandomize_channel(channel, blk_shape=(8, 8)):
    rows = channel.shape[0] // blk_shape[0] * blk_shape[0]
    cols = channel.shape[1] // blk_shape[1] * blk_shape[1]
    blks = np.array([[
        channel[i:i + blk_shape[0], j:j + blk_shape[1]]
        for j in range(0, cols, blk_shape[1])
    ] for i in range(0, rows, blk_shape[0])])
    shape = blks.shape
    blks = blks.reshape(-1, blk_shape[0], blk_shape[1])
    blk_num = blks.shape[0]
    indices = np.arange(blk_num)
    np.random.RandomState(0).shuffle(indices)
    res = np.zeros(blks.shape)
    res[indices] = blks
    res = np.concatenate(np.concatenate(res.reshape(shape), 1), 1)
    full_res = np.copy(channel)
    full_res[:rows, :cols] = res
    return full_res


def derandomize_img(img):
    for i in range(3):
        img[:, :, i] = derandomize_channel(img[:, :, i])
    return img


def rebin(a, shape):
    if a.shape[0] % 2 == 1:
        a = np.vstack((a, np.zeros((1, a.shape[1]))))
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


if __name__ == '__main__':
	wm = np.zeros((134, 240))
	for i in range(31,61):
		img = cv2.imread("../output/extracted/frame{}.jpeg".format(i), cv2.IMREAD_GRAYSCALE).astype(np.float32)
		wm += img
	wm = wm * 1.0 / np.amax(wm) * 255.0
	# wm *= 1 / 30.0
	cv2.imwrite("../output/res.jpeg", wm.astype(np.uint8))
