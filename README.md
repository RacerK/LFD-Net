# LFD-Net

#### Lightweight Feature-interaction Dehazing Network for Real-time Vision Tasks

![](readme_images/framework.png)

##### Dependencies

This code is developed using Anaconda python 3.6.6. To install the necessary dependencies for this code, run the following command:

```bash
pip install -r requirements.txt
```

This will install all the required dependencies for the code to run properly, as follows:

```textile
torch==1.10.2
torchvision==0.11.3
PIL==9.4.0
matplotlib==3.3.4
opencv==4.6.0
numpy=1.19.5
```

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
