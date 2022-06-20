import numpy as np
import copy
import cv2
import pywt
import math
import time
from utils import randomize_channel, derandomize_channel

class DwtWatermarkEncoder:
    def __init__(self, scales=[0.01, 0.0025, 0.0025, 0.0025]):
        self.scales = scales

    def read_img(self, path):
        img = cv2.imread(path)
        assert img is not None, "Image not found in {}".format(path)
        self.img = img.astype(np.float32)
        self.wm_shape = (int(img.shape[1] / 2), int(img.shape[0] / 2))

    def read_wm(self, path):
        wm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert wm is not None, "Watermark not found in {}".format(path)
        wm = cv2.resize(wm, self.wm_shape, interpolation = cv2.INTER_AREA)
        self.wm = (wm > 127).astype(np.uint8) * 255

    def watermark_coeff(self, coeff, i):
        coeff += self.scales[i] * self.wm

    def encode(self, img):
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        u = yuv[:,:,1]
        u = randomize_channel(u)
        LL, (LH, HL, HH) = pywt.dwt2(u, 'haar')
        self.watermark_coeff(LL, 0)
        self.watermark_coeff(LH, 1)
        self.watermark_coeff(HL, 2)
        self.watermark_coeff(HH, 3)
        wmed_u = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
        yuv[:,:,1] = derandomize_channel(wmed_u)
        wmed_img = cv2.cvtColor(yuv, cv2.COLOR_YCR_CB2BGR)
        return wmed_img

    def embed(self, output_path):
        wmed_img = self.encode(self.img)
        cv2.imwrite(output_path, wmed_img)
        return wmed_img

    def embed_video(self, wm_path, video_path, output_path):
        frame_dim = (1920, 1080)
        resized_dim = (int(frame_dim[0] / 2), int(frame_dim[1] / 2))
        self.wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
        self.wm = cv2.resize(self.wm, resized_dim, interpolation = cv2.INTER_AREA)
        self.wm = (self.wm > 127).astype(np.uint8) * 255
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
        wmed_img_yuv = cv2.cvtColor(wmed_img, cv2.COLOR_BGR2YCR_CB)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        wmed_u = randomize_channel(wmed_img_yuv[:,:,1])
        u = randomize_channel(img_yuv[:,:,1])
        LLw, (LHw, HLw, HHw) = pywt.dwt2(wmed_u, 'haar')
        LL, (LH, HL, HH) = pywt.dwt2(u, 'haar')
        wm_LL = (LLw - LL) / self.scales[0]
        return wm_LL

    def extract(self, img_path, wmed_img_path, output_path):
        img = cv2.imread(img_path).astype(np.float32)
        wmed_img = cv2.imread(wmed_img_path).astype(np.float32)
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
    # # Image Watermarking
    # img_path = "../pics/imgs/frame63.jpeg"
    # wm_path = "../pics/wmks/wmk2.jpeg"
    # output_path = "../output/watermarked.jpeg"
    # extracted_path = "../output/extracted_watermark.jpeg"
    # encoder = DwtWatermarkEncoder()
    # encoder.read_img(img_path)
    # encoder.read_wm(wm_path)
    # encoder.embed(output_path)
    # DwtWatermarkDecoder().extract(img_path, output_path, extracted_path)

    # Video Watermarking
    video_path = '../videos/bbb-short.mp4'
    wm_path = "../pics/wmks/wmk2.jpeg"
    output_path = '../output/watermarked.mp4'
    extracted_path = "../output/extracted"
    start_time = time.time()
    encoder = DwtWatermarkEncoder()
    encoder.embed_video(wm_path, video_path, output_path)
    print("Encoding time: {}s".format(time.time() - start_time))
    start_time = time.time()
    decoder = DwtWatermarkDecoder()
    decoder.extract_video(video_path, output_path, extracted_path)
    print("Decoding time: {}s".format(time.time() - start_time))
