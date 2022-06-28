import cv2
import dtcwt
import numpy as np
import time
from utils import rebin

step = 5
alpha = 0.8

class DwtWatermarkEncoder:
	def __init__(self, scales=[0.01, 0.0025, 0.0025, 0.0025]):
		self.scales = scales

	def encode(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		wm_transform = dtcwt.Transform2d()
		wm_coeffs = wm_transform.forward(self.wm, nlevels=1)
		img_transform = dtcwt.Transform2d()
		img_coeffs = img_transform.forward(img[:,:,1], nlevels=3)
		y_transform = dtcwt.Transform2d()
		y_coeffs = y_transform.forward(img[:,:,1], nlevels=3)

		# Masks for the level 3 subbands
		masks3 = [0 for i in range(6)]
		shape3 = y_coeffs.highpasses[2][:,:,0].shape
		for i in range(6):
			masks3[i] = np.ceil(rebin(np.abs(y_coeffs.highpasses[1][:,:,i]), shape3) * (1 / step))
			masks3[i] *= 1.0 / np.amax(masks3[i])
			# cv2.imwrite("../output/{}.jpeg".format(i), masks3[i].astype(np.uint8))
		# masks4 = [0 for i in range(6)]
		# for i in range(6):
		# 	masks4[i] = np.ceil(rebin(masks3[i], ((shape3[0] + 1)//2, (shape3[1] + 1)//2)) * (1 / 0.05))
		# beta = 0.001
		for i in range(6):
			coeff = wm_coeffs.highpasses[0][:,:,i]
			coeffs = np.vstack((np.hstack((coeff[:-1,:], coeff[:-1,:])), np.hstack((coeff, coeff))))
			img_coeffs.highpasses[2][:,:,i] += alpha * (masks3[i] * coeffs)
			# img_coeffs.highpasses[3][:,:,i] += beta * (masks4[i] * coeff)
		img[:,:,1] = img_transform.inverse(img_coeffs)
		wmed_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
		return wmed_img

	def embed(self, wm_path, input_path, output_path):
		img = cv2.imread(input_path)
		assert img is not None, "Image not found in {}".format(input_path)
		self.img = img.astype(np.float32)

		wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
		assert wm is not None, "Watermark not found in {}".format(wm_path)
		wm = cv2.resize(wm, (img.shape[1] // 8, img.shape[0] // 8), interpolation = cv2.INTER_AREA)
		self.wm = (wm > 127).astype(np.uint8) * 255
		self.wm = self.wm.astype(np.int32)
		self.wm[self.wm != 255] = -255

		wmed_img = self.encode(self.img)
		cv2.imwrite(output_path, wmed_img)
		return wmed_img

	def embed_video(self, wm_path, video_path, output_path):
		frame_dim = (1920, 1080)
		resized_dim = (int(frame_dim[0] / 2), int(frame_dim[1] / 2))
		self.wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
		self.wm = cv2.resize(self.wm, (frame_dim[0] // 8, frame_dim[1] // 8), interpolation = cv2.INTER_AREA)
		self.wm = (self.wm > 127).astype(np.uint8) * 255
		self.wm = self.wm.astype(np.int32)
		self.wm[self.wm != 255] = -255
		cap = cv2.VideoCapture(video_path)
		fourcc = cv2.VideoWriter_fourcc(*'avc1')
		out = cv2.VideoWriter(output_path, fourcc, 30, frame_dim)
		count = 0
		while cap.isOpened():
			ret, frame = cap.read()
			if ret:
				count += 1
				frame = frame.astype(np.float32)
				print("Start processing frame {}".format(count))
				wmed_frame = self.encode(frame)
				wmed_frame = np.clip(wmed_frame, a_min=0, a_max=255)
				out.write(wmed_frame.astype(np.uint8))
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			else:
				break

class DwtWatermarkDecoder:
	def __init__(self, scales=[0.01, 0.0025, 0.0025, 0.0025]):
		self.scales = scales

	def decode(self, img, wmed_img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		wmed_img = cv2.cvtColor(wmed_img, cv2.COLOR_BGR2YUV)
		
		img_transform = dtcwt.Transform2d()
		img_coeffs = img_transform.forward(img[:,:,1], nlevels=3)
		wmed_transform = dtcwt.Transform2d()
		wmed_coeffs = wmed_transform.forward(wmed_img[:,:,1], nlevels=3)
		y_transform = dtcwt.Transform2d()
		y_coeffs = y_transform.forward(img[:,:,1], nlevels=3)

		masks3 = [0 for i in range(6)]
		inv_masks3 = [0 for i in range(6)]
		shape3 = y_coeffs.highpasses[2][:,:,0].shape
		for i in range(6):
			masks3[i] = np.ceil(rebin(np.abs(y_coeffs.highpasses[1][:,:,i]), shape3) * (1.0 / step))
			masks3[i][masks3[i] == 0] = 0.01
			masks3[i] *= 1.0 / np.amax(masks3[i])
			inv_masks3[i] = 1.0 / masks3[i]

		# beta = 0.001
		# inv_masks4 = [0 for i in range(6)]
		# for i in range(6):
		# 	inv_masks4[i] = np.ceil(rebin(masks3[i], ((shape3[0] + 1)//2, (shape3[1] + 1)//2)) * (1 / 0.05))
		# 	inv_masks4[i] = 1.0 / inv_masks4[i]

		shape = wmed_coeffs.highpasses[2][:,:,i].shape
		shape = (shape[0] // 2, shape[1] // 2)
		coeffs = np.zeros((shape[0], shape[1], 6), dtype = 'complex_')
		for i in range(6):
			coeff = (wmed_coeffs.highpasses[2][:,:,i]) * inv_masks3[i] * 1 / alpha
			# c = (wmed_coeffs.highpasses[3][:,:,i]) * inv_masks4[i] * 1 / beta
			coeffs[:,:,i] = coeff[:shape[0],:shape[1]] + coeff[:shape[0],shape[1]:] + coeff[shape[0]:-1,:shape[1]] + coeff[shape[0]:-1,shape[1]:]
		highpasses = tuple([coeffs])
		lowpass = np.zeros((shape[0] * 2, shape[1] * 2))
		t = dtcwt.Transform2d()
		wm = t.inverse(dtcwt.Pyramid(lowpass, highpasses))
		# wm = (wm > 100).astype(np.uint8) * 255
		return wm

	def extract(self, img_path, wmed_img_path, output_path):
		img = cv2.imread(img_path).astype(np.float32)
		wmed_img = cv2.imread(wmed_img_path).astype(np.float32)
		wmed_img = cv2.resize(wmed_img, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)
		wm = self.decode(img, wmed_img)
		cv2.imwrite(output_path, wm)

	def extract_video(self, original_video_path, wmed_video_path, output_folder):
		wmed_cap = cv2.VideoCapture(wmed_video_path)
		cap = cv2.VideoCapture(original_video_path)
		count = 0
		while wmed_cap.isOpened():
			ret, wmed_frame = wmed_cap.read()
			if ret:
				if cap.isOpened():
					ret, frame = cap.read()
					if ret:
						wmed_frame = cv2.resize(wmed_frame, (frame.shape[1], frame.shape[0]))
						count += 1
						wmed_frame = wmed_frame.astype(np.float32)
						frame = frame.astype(np.float32)
						print("Start processing frame {}".format(count))
						wm = self.decode(frame, wmed_frame)
						cv2.imwrite('{}/frame{}.jpeg'.format(output_folder, count), wm)
						if cv2.waitKey(1) & 0xFF == ord('q'):
							break
			else:
				break

if __name__ == '__main__':
	# img_path = "../pics/imgs/frame63.jpeg"
	# wm_path = "../pics/wmks/wmk5.jpg"
	# output_path = "../output/watermarked.jpg"
	# extracted_path = "../output/extracted_watermark.jpeg"
	# DwtWatermarkEncoder().embed(wm_path, img_path, output_path)
	# DwtWatermarkDecoder().extract(img_path, output_path, extracted_path)
	wm_path = "../pics/wmks/wmk5.jpg"
	video_path = "../videos/bbb-short.mp4"
	output_path = "../output/output.mp4"
	extracted_path = "../output/extracted"
	start_time = time.time()
	DwtWatermarkEncoder().embed_video(wm_path, video_path, output_path)
	print("Encoding time: {}s", time.time() - start_time)
	DwtWatermarkDecoder().extract_video(video_path, output_path, extracted_path)
