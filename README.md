# video-watermark
## Testing Algorithms
> * **DWT-DCT-SVD**: This is derived from [blind_watermark](https://github.com/guofei9987/blind_watermark) and [invisible-watermark](https://github.com/ShieldMnt/invisible-watermark) and is applied frame by frame on videos.
> 
> * **DWT**: This is inspired by [A chaos-based robust wavelet-domain watermarking algorithm](http://isrc.ccs.asia.edu.tw/yourslides/files/20/0915.pdf).
## Usage
Please ensure paths in ```dwt_dct_svd_wm.py``` and ```dwt_wm.py``` exist before usage.
```
cd src
python3 dwt_dct_svd_wm.py
python3 dwt_wm.py
```
## Experimental Results
Check the videos in the `output` folder
> * **DWT-DCT-SVD**: 500ms per frame (not optimized), better imperceptibility, poor recovery of watermark, can't recover watermarks after transcoding from H.264 to H.265
> * **DWT**: 100ms per frame (not optimized), acceptable imperceptibility, watermark only recoverable in frames with less motion after transcoding from H.264 to H.265 