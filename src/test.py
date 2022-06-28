import cv2
import dtcwt
import numpy as np
from utils import rebin, randomize_channel, derandomize_channel

scales = [0.008, 0.005]

class DwtWatermarkEncoder:
    def __init__(self, scales=[0.01, 0.0025, 0.0025, 0.0025]):
        self.scales = scales

    def encode(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img[:,:,1] = randomize_channel(img[:,:,1])
        wm_transform = dtcwt.Transform2d()
        wm_coeffs = wm_transform.forward(self.wm, nlevels=1)
        img_transform = dtcwt.Transform2d()
        img_coeffs = img_transform.forward(img[:,:,1], nlevels=1)
        for i in range(6):
            img_coeffs.highpasses[0][:,:,i] += wm_coeffs.highpasses[0][:,:,i] * scales[1]
        img_coeffs.lowpass += wm_coeffs.lowpass * scales[0]
        img[:,:,1] = img_transform.inverse(img_coeffs)

        img[:,:,1] = derandomize_channel(img[:,:,1])
        wmed_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        return wmed_img

    def embed(self, wm_path, input_path, output_path):
        img = cv2.imread(input_path)
        assert img is not None, "Image not found in {}".format(input_path)
        self.img = img.astype(np.float32)

        wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
        assert wm is not None, "Watermark not found in {}".format(wm_path)
        wm = cv2.resize(wm, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)
        self.wm = (wm > 127).astype(np.uint8) * 255
        self.wm = self.wm.astype(np.int32)
        # self.wm[self.wm != 255] = -255

        wmed_img = self.encode(self.img)
        cv2.imwrite(output_path, wmed_img)
        return wmed_img

    def embed_video(self, wm_path, video_path, output_path):
        frame_dim = (1920, 1080)
        self.wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
        self.wm = cv2.resize(self.wm, (frame_dim[0], frame_dim[1]), interpolation = cv2.INTER_AREA)
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
        img[:,:,1] = randomize_channel(img[:,:,1])
        wmed_img[:,:,1] = randomize_channel(wmed_img[:,:,1])
        
        img_transform = dtcwt.Transform2d()
        img_coeffs = img_transform.forward(img[:,:,1], nlevels=1)
        wmed_transform = dtcwt.Transform2d()
        wmed_coeffs = wmed_transform.forward(wmed_img[:,:,1], nlevels=1)

        shape = wmed_coeffs.highpasses[0][:,:,0].shape
        coeffs = np.zeros((shape[0], shape[1], 6), dtype = 'complex_')
        for i in range(6):
            coeffs[:,:,i] = wmed_coeffs.highpasses[0][:,:,i] - img_coeffs.highpasses[0][:,:,i]
            coeffs[:,:,i] = coeffs[:,:,i] / scales[1]
        highpasses = tuple([coeffs])
        lowpass = (wmed_coeffs.lowpass - img_coeffs.lowpass) / scales[0]
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
    img_path = "../pics/imgs/frame63.jpeg"
    wm_path = "../pics/wmks/wmk5.jpg"
    output_path = "../output/watermarked.jpg"
    extracted_path = "../output/extracted_watermark.jpeg"
    DwtWatermarkEncoder().embed(wm_path, img_path, output_path)
    DwtWatermarkDecoder().extract(img_path, output_path, extracted_path)
    # wm_path = "../pics/wmks/wmk4.jpg"
    # video_path = "../videos/bbb-short.mp4"
    # output_path = "../output/output_720p.mp4"
    # extracted_path = "../output/extracted"
    # DwtWatermarkEncoder().embed_video(wm_path, video_path, output_path)
    # DwtWatermarkDecoder().extract_video(video_path, output_path, extracted_path)
