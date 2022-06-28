# video-watermark
## Testing Algorithms
> * **DWT-DCT-SVD**: Derived from [blind_watermark](https://github.com/guofei9987/blind_watermark) and [invisible-watermark](https://github.com/ShieldMnt/invisible-watermark).
> * **DWT**: Inspired by [A chaos-based robust wavelet-domain watermarking algorithm](http://isrc.ccs.asia.edu.tw/yourslides/files/20/0915.pdf).
> * **DT-CWT**: Inspired by [Imperceptible and Robust Blind Video Watermarking Using Chrominance Embedding: A Set of Approaches in the DT CWT Domain](https://www.researchgate.net/profile/Md-Asikuzzaman/publication/264673001_Imperceptible_and_Robust_Blind_Video_Watermarking_Using_Chrominance_Embedding_A_Set_of_Approaches_in_the_DT_CWT_Domain/links/54ae3dbe0cf2213c5fe42ae9/Imperceptible-and-Robust-Blind-Video-Watermarking-Using-Chrominance-Embedding-A-Set-of-Approaches-in-the-DT-CWT-Domain.pdf).
## Usage
Please ensure paths in ```dwt_dct_svd_wm.py``` and ```dwt_wm.py``` exist before usage.
```
cd src
python3 dwt_dct_svd_wm.py
python3 dwt_wm.py
python3 dtcwt_wm.py
```
## Experimental Results
Check the videos in the `output` folder
> * **DWT-DCT-SVD**: blind, 500ms per frame (not optimized), good imperceptibility, poor recovery of watermark, can't recover watermarks after transcoding from H.264 to H.265
> * **DWT**: non-blind, 100ms per frame (not optimized), acceptable imperceptibility, watermark only recoverable in frames with less motion after transcoding from H.264 to H.265, not robost against downscaling
> * **DT-CWT**: blind, 300ms per frame (not optimized), good imperceptibility, robust against downscaling, robust against transcoding from H.264 to H.265
