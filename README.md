# LFD-Net

#### Lightweight Feature-interaction Dehazing Network

![](framework.png)

##### Training

```bash
python train.py -th haze_images/ -to original_images/ -e 50 -lr 0.001
```

##### Inferencing

Place all the hazy images that need to be dehazed in the test directory and specify a destination directory for the dehazed images.

```bash
python infer_multi.py -td hazy/ -ts dehaze/ 
```

##### Evaluation

Four evaluation metrics will be applied to all dehazed and ground truth image pairs: PSNR, SSIM, CIEDE2000, and SSEQ. The results of these metrics will be written to `metrics.txt`.

```bash
python evaluate.py -to gt/ -td dehaze/
```