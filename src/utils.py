import cv2
import numpy as np

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

def randomize_channel(channel):
	blk_size = 8
	blks = [[channel[i:i + blk_size, j:j + blk_size] for j in range(0, channel.shape[1], blk_size)] for i in range(0, channel.shape[0], blk_size)]
	blks = np.array(blks)
	shape = blks.shape
	blks = blks.reshape(-1, blk_size, blk_size)
	np.random.RandomState(0).shuffle(blks)
	res = np.concatenate(np.concatenate(blks.reshape(shape), 1), 1)
	return res

def randomize_img(img):
	for i in range(3):
		img[:,:,i] = randomize_channel(img[:,:,i])
	return img

def derandomize_channel(channel):
	blk_size = 8
	blks = [[channel[i:i + blk_size, j:j + blk_size] for j in range(0, channel.shape[1], blk_size)] for i in range(0, channel.shape[0], blk_size)]
	blks = np.array(blks)
	shape = blks.shape
	blks = blks.reshape(-1, blk_size, blk_size)
	blk_num = blks.shape[0]
	indices = np.arange(blk_num)
	np.random.RandomState(0).shuffle(indices)
	res = np.zeros(blks.shape)
	res[indices] = blks
	res = np.concatenate(np.concatenate(res.reshape(shape), 1), 1)
	return res

def derandomize_img(img):
	for i in range(3):
		img[:,:,i] = derandomize_channel(img[:,:,i])
	return img

if __name__ == '__main__':
	img = cv2.imread('../pics/wmks/wmk1.jpg', cv2.IMREAD_GRAYSCALE)
	at = ArnoldTransform(img)
	for i in range(193):
		at.transform()
		cv2.imwrite('../output/img{}.jpeg'.format(at.count), at.img)
