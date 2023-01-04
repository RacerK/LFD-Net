import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import io, color
import os
import argparse
from scipy.stats import entropy


def rmetrics(corrected, reference):
    height, width, channels = reference.shape
    psnr = compare_psnr(corrected, reference)
    ssim = compare_ssim(corrected, reference, win_size=11, data_range=255, multichannel=True)

    clear_lab = color.rgb2lab(reference)
    dehaze_lab = color.rgb2lab(corrected)

    difference = color.deltaE_ciede2000(dehaze_lab, clear_lab)
    ciede = sum(sum(difference)) / height / width

    spatial_component = corrected[:, :, 0]             # R channel is the spatial component
    spectral_component = corrected[:, :, 1].flatten()  # G and B channels are the spectral component
    spatial_entropy = entropy(spatial_component)
    spectral_entropy = entropy(spectral_component)
    SSEQ_dehaze = spatial_entropy + spectral_entropy 

    spatial_component = reference[:, :, 0]             # R channel is the spatial component
    spectral_component = reference[:, :, 1].flatten()  # G and B channels are the spectral component
    spatial_entropy = entropy(spatial_component)
    spectral_entropy = entropy(spectral_component)
    SSEQ_clear = spatial_entropy + spectral_entropy   

    sseq = abs(sum(SSEQ_dehaze - SSEQ_clear) / sum(SSEQ_clear))
    if np.isinf(sseq) or np.isnan(sseq):
        sseq = 1
    
    return psnr, ssim, ciede, sseq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-to", "--test_original", required=True, help="path to original testing images")
    ap.add_argument("-td", "--test_dehaze", required=True, help="path to dehazed testing images")
    args = vars(ap.parse_args())

    result_path = args["test_dehaze"]
    reference_path = args["test_original"]

    result_dirs = os.listdir(result_path)
    sum_psnr, sum_ssim, sum_ciede, sum_sseq = 0., 0., 0., 0.

    cnt = 0

    for image_name in result_dirs:
        cnt += 1
        try: 
            corrected = io.imread(os.path.join(result_path, image_name))
            reference = io.imread(os.path.join(reference_path, image_name))
        except Exception:
            continue
    
        psnr, ssim, ciede, sseq = rmetrics(corrected, reference)
        
        sum_psnr += psnr
        sum_ssim += ssim
        sum_ciede += ciede
        sum_sseq += sseq

        print('{}: psnr={} ssim={} cie={} sseq={} \n'.format(image_name, psnr, ssim, ciede, sseq))
        with open(os.path.join(result_path, 'metrics.txt'), 'a') as f:
            f.write('{}: psnr={} ssim={} cie={} sseq={} \n'.format(image_name, psnr, ssim, ciede, sseq))

    mean_psnr = sum_psnr / cnt
    mean_ssim = sum_ssim / cnt
    mean_ciede = sum_ciede / cnt
    mean_sseq = sum_sseq / cnt
    
    with open(os.path.join(result_path,'metrics.txt'), 'a') as f:
        f.write('Average: psnr={} ssim={} cie={} sseq={} \n'.format(mean_psnr, mean_ssim, mean_ciede, mean_sseq))
    print('Average: psnr={} ssim={} cie={} sseq={} \n'.format(mean_psnr, mean_ssim, mean_ciede, mean_sseq))
    f.close()


if __name__ == '__main__':
    main()
