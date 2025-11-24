````markdown
## Image Denoising Project

This project compares classical filtering and modern deep learning models on grayscale images.

## Methods

### Classical
- Smoothing filter  
- Median filter  
- Adaptive (Lee) filter  

### Deep Learning
- DnCNN [Zhang et al., CVPR 2017]  
- Restormer [Zamir et al., CVPR 2022]  

## Why These Models

- **DnCNN**  
  - Simple residual CNN.  
  - Strong baseline for Gaussian denoising with blind noise level.  
  - Widely used reference model, so your results are easy to compare with prior work.

- **Restormer**  
  - Transformer based architecture with large receptive field and efficient attention.  
  - Designed for high resolution restoration and strong at heavy noise.  
  - Represents a recent state of the art model, so you can show the gap between classical, CNN, and transformer methods.

## Input Dataset

- Four 512×512 grayscale images.  
- Mix of textures and structures:
  - One face portrait.  
  - One natural landscape.  
  - One urban street scene.  
  - One generic benchmark image.  
- Synthetic noise types:
  - Gaussian noise with different σ levels (15, 25, 50).  
  - Salt and pepper noise at different densities (2%, 5%).  

This setup gives you:
- Smooth regions, edges, and fine textures.  
- Both flat backgrounds and complex high frequency areas.  
- A small but diverse testbed to visualize failures and strengths of each method.

## Workflow

1. **Preprocess images**
```bash
python preprocessing.py
````

2. **Add noise (Gaussian, salt and pepper)**

```bash
python noise_maker.py
```

3. **Run classical denoisers**

```bash
python denoise_classical.py
```

4. **Run deep learning models**

```bash
python denoise_dncnn.py
python denoise_restormer.py
```

## Folder Structure

```text
assets/
  lenna.png
  nature.png
  portrait.png
  street.png
  noise/
  output_smoothing/
  output_median/
  output_adaptive/
  output_dncnn/
  output_restormer/
  weights/
    dncnn25.pth
    gaussian_gray_denoising_sigma50.pth   # Restormer
metrics/
  lenna_dncnn.json
  lenna_restormer.json
  ...
```

## References

[1] K. Zhang, W. Zuo, Y. Chen, D. Meng, L. Zhang,
"Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising,"
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[2] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M. H. Yang,
"Restormer: Efficient Transformer for High Resolution Image Restoration,"
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

```
::contentReference[oaicite:0]{index=0}
```
