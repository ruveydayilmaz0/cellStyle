import numpy as np
import glob
from skimage import io
from pathlib import Path


# Load the images first
root = "/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2024/BF-C2DL-HSC/"
root_images = root + "01/"
root_semantics = root + "01_ST/SEG/"
save_path = "/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2024/BF-C2DL-HSC_backRemoved/01/"
Path(save_path).mkdir(exist_ok=True, parents=True)
img_paths = sorted(glob.glob(root_images + '*.tif'))

# Parameters are for background from the style image (CTC_huh7)
mean = 3
std_dev = 1

for img_path in img_paths:
    img = io.imread(img_path)
    img_name = img_path.split('/')[-1]
    try:
        mask = io.imread(root_semantics + 'man_seg' + img_name[1:])
        # mask = io.imread(root_semantics + 'mask_' + img_name)
    except:
        continue
    background = np.clip(np.random.normal(mean, std_dev, (img.shape)), 1, 10).astype(np.uint8)
    background[mask>0] = img[mask>0]
    io.imsave(save_path + img_name, background)