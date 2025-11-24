## Image Denoising Project

This project compares classical and deep learning denoising methods on grayscale images.

## Methods
## Classical
- Smoothing filter  
- Median filter  
- Adaptive (Lee) filter  

## Deep Learning
- DnCNN  
- SwinIR  
- Diffusion model (optional)  

## Workflow
1. Preprocess images  
```

python preprocessing.py

```

2. Add noise (Gaussian, salt-pepper)  
```

python noise_maker.py

```

3. Run classical denoisers  
```

python denoise_classical.py

```

4. Run deep learning models  
```

python denoise_dncnn.py
python denoise_swinir.py
python denoise_ddpm.py   

```

## Folder Structure
```

assets/
noise/
output_smoothing/
output_median/
output_adaptive/
output_dncnn/
output_swinir/
output_ddpm/
weights/
dncnn25.pth
swinir_gray25.pth

```

All scripts operate on 512×512 grayscale images. Outputs are saved in their respective folders.

```



