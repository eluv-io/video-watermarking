import numpy as np
import copy
import cv2
import pywt
import math

class DwtDctSvdEncoder:
    def __init__(self, wm, scales=[0,36,0], blk=4):
        self.wm = wm
        self.scales = scales
        self.blk = blk

    def encode(self, bgr):
        (row, col, channels) = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        for channel in range(3):
            if self.scales[channel] <= 0:
                continue
            ca, hvd = pywt.dwt2(yuv[:row // 4 * 4,:col // 4 * 4, channel], 'haar')
            self.encode_frame(ca, self.scales[channel])
            yuv[:row // 4 * 4, :col // 4 * 4, channel] = pywt.idwt2((ca, hvd), 'haar')
        bgr_encoded = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return bgr_encoded

    def encode_frame(self, frame, scale):
        (row, col) = frame.shape
        c = 0
        for i in range(row // self.blk):
            for j in range(col // self.blk):
                blk = frame[i * self.blk : i * self.blk + self.blk,
                              j * self.blk : j * self.blk + self.blk]
                wm_bit = self.wm[c % self.wm.size]
                embedded_blk = self.blk_embed_wm(blk, wm_bit, scale)
                frame[i * self.blk : i * self.blk + self.blk,
                      j * self.blk : j * self.blk + self.blk] = embedded_blk
                c += 1

    def blk_embed_wm(self, blk, wm_bit, scale):
        u, s, v = np.linalg.svd(cv2.dct(blk))
        s[0] = (s[0] // scale + 0.25 + 0.5 * wm_bit) * scale
        return cv2.idct(np.dot(u, np.dot(np.diag(s), v)))

class DwtDctSvdDecoder:
    def __init__(self, wm_size, scales=[0,36,0], blk=4):
        self.wm_size = wm_size
        self.scales = scales
        self.blk = blk

    def decode(self, bgr):
        (row, col, channels) = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        self.block_num = row * col // 4 // (self.blk * self.blk)
        wm_bits = np.zeros(shape=(3, self.block_num))
        for channel in range(3):
            if self.scales[channel] <= 0:
                continue
            ca, hvd = pywt.dwt2(yuv[:row // 4 * 4,:col // 4 * 4, channel], 'haar')
            self.decode_frame(ca, self.scales[channel], wm_bits[channel])
        # Average 3 channels
        wm_avg = np.zeros(shape=self.wm_size)
        for i in range(self.wm_size):
            wm_avg[i] = wm_bits[1, i::self.wm_size].mean()
        return wm_avg

    def decode_frame(self, frame, scale, wm_bits):
        (row, col) = frame.shape
        c = 0
        for i in range(row // self.blk):
            for j in range(col // self.blk):
                blk = frame[i * self.blk : i * self.blk + self.blk,
                            j * self.blk : j * self.blk + self.blk]
                wm_bit = self.blk_extract_wm(blk, scale)
                wm_bits[c] = wm_bit
                c += 1

    def blk_extract_wm(self, blk, scale):
        u,s,v = np.linalg.svd(cv2.dct(blk))
        wm = int((s[0] % scale) > scale * 0.5)
        return wm
