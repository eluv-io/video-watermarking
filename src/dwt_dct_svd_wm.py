import cv2
import numpy as np
import time
from dwt_dct_svd import *

class WaterMarkEncoder:
	def __init__(self, pwd_wm=0):
		self.pwd_wm = pwd_wm

	def maximum_wm_size(self):
		assert self.img is not None
		row, col, channels = self.img.shape
		block_num = row * col // 64
		print("Maximum watermark size: {}KB".format(block_num / 1000))

	def read_img(self, path):
		img = cv2.imread(path)
		assert img is not None, "Image not found in {}".format(path)
		self.img = img.astype(np.float32)

	def read_wm(self, path):
		wm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		assert wm is not None, "Watermark not found in {}".format(path)
		self.wm_bits = wm.flatten() > 128
		np.random.RandomState(self.pwd_wm).shuffle(self.wm_bits)

	def embed(self, output_path):
		wmed_img = DwtDctSvdEncoder(self.wm_bits).encode(self.img)
		cv2.imwrite(output_path, wmed_img)
		return wmed_img

class WaterMarkDecoder:
	def __init__(self, wm_shape=(128, 128), pwd_wm=0):
		self.pwd_wm = pwd_wm
		self.wm_shape = wm_shape
		self.wm_size = np.array(wm_shape).prod()

	def extract(self, input_path, output_path):
		img = cv2.imread(input_path)
		assert img is not None, "Wat ermarked image not found in {}".format(input_path)
		self.img = img.astype(np.float32)
		wm_bits = DwtDctSvdDecoder(self.wm_size).decode(self.img)
		wm_idx = np.arange(self.wm_size)
		np.random.RandomState(self.pwd_wm).shuffle(wm_idx)
		wm_bits[wm_idx] = wm_bits.copy()
		wm = wm_bits.reshape(self.wm_shape) * 255
		cv2.imwrite(output_path, wm)
		return wm

class WaterMarkVideoEncoder:
	def __init__(self, pwd_wm=0):
		self.pwd_wm = pwd_wm

	def read_wm(self, path):
		wm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		assert wm is not None, "Watermark not found in {}".format(path)
		self.wm_bits = wm.flatten() > 128
		np.random.RandomState(self.pwd_wm).shuffle(self.wm_bits)

	def embed(self, video_path, output_path):
		cap = cv2.VideoCapture(video_path)
		fourcc = cv2.VideoWriter_fourcc(*'avc1')
		out = cv2.VideoWriter(output_path, fourcc, 30, (1920, 1080))
		encoder = DwtDctSvdEncoder(self.wm_bits)
		count = 0
		while cap.isOpened():
			ret, frame = cap.read()
			if ret:
				count += 1
				print("Start processing frame {}".format(count))
				wmed_frame = encoder.encode(frame.astype(np.float32))
				wmed_frame = np.clip(wmed_frame, a_min=0, a_max=255)
				cv2.imwrite("../output/q/frame{}.jpeg".format(count), wmed_frame.astype(np.uint8))
				# print(wmed_frame.astype(np.uint8).shape)
				out.write(wmed_frame.astype(np.uint8))
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			else:
				break
		cap.release()
		out.release()

class WaterMarkVideoDecoder:
	def __init__(self, wm_shape=(128, 128), pwd_wm=0):
		self.pwd_wm = pwd_wm
		self.wm_shape = wm_shape
		self.wm_size = np.array(wm_shape).prod()

	def extract(self, wmed_video_path, output_path):
		wmed_cap = cv2.VideoCapture(wmed_video_path)
		decoder = DwtDctSvdDecoder(self.wm_size)
		wm_idx = np.arange(self.wm_size)
		np.random.RandomState(self.pwd_wm).shuffle(wm_idx)
		count = 0
		while wmed_cap.isOpened():
			ret, wmed_frame = wmed_cap.read()
			if ret:
				count += 1
				print("Start processing frame {}".format(count))
				wm_bits = decoder.decode(wmed_frame.astype(np.float32))
				wm_bits[wm_idx] = wm_bits.copy()
				wm = wm_bits.reshape(self.wm_shape) * 255
				cv2.imwrite('{}/frame{}.jpeg'.format(output_path, count), wm)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			else:
				break

if __name__ == '__main__':
	# # Image Watermarking
	# img_path = '../pics/imgs/frame63.jpeg'
	# wm_path = '../pics/wmks/wmk3.jpg'
	# wmed_img_path = '../output/output.jpg'
	# output_path = '../output/extracted.jpg'

	# wm_encoder = WaterMarkEncoder()
	# wm_encoder.read_img(img_path)
	# wm_encoder.maximum_wm_size()
	# wm_encoder.read_wm(wm_path)
	# start_time = time.time()
	# wm_encoder.embed(wmed_img_path)
	# print("Encoding time: {}s".format(time.time() - start_time))

	# start_time = time.time()
	# wm_decoder = WaterMarkDecoder(wm_shape=(200,200))
	# wm_decoder.extract(wmed_img_path, output_path)
	# print("Decoding time: {}s".format(time.time() - start_time))

	# Video Watermarking
	video_path = "../videos/bbb-short.mp4"
	wm_path = "../pics/wmks/wmk3.jpg"
	output_path = "../output/output.mp4"
	output_folder = "../output/extracted"

	wm_encoder = WaterMarkVideoEncoder()
	wm_encoder.read_wm(wm_path)
	start_time = time.time()
	wm_encoder.embed(video_path, output_path)
	print("Encoding time: {}s".format(time.time() - start_time))

	start_time = time.time()
	wm_decoder = WaterMarkVideoDecoder(wm_shape=(200, 200))
	wm_decoder.extract(output_path, output_folder)
	print("Decoding time: {}s".format(time.time() - start_time))
