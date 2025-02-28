import tifffile as tiff
from pathlib import Path
import glob

# Load the 3D image from the TIFF file
files_dir = "/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2024/Fluo-C3DL-MDA231/"
img_paths = sorted(glob.glob(files_dir + "01/*.tif"))
mask_paths = sorted(glob.glob(files_dir + "01_ST/SEG/*.tif"))
out_path = "/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2024/Fluo-C3DL-MDA231_slice9/"
Path(out_path + "01").mkdir(exist_ok=True, parents=True)
Path(out_path + "01_ST/SEG").mkdir(exist_ok=True, parents=True)

for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):

    image = tiff.imread(img_path)
    im_slice = image[8]
    mask = tiff.imread(mask_path)
    mask_slice = mask[8]

    tiff.imwrite(out_path + "01/t" + str(i).zfill(3) + ".tif", im_slice)
    tiff.imwrite(out_path + "01_ST/SEG/man_seg" + str(i).zfill(3) + ".tif", mask_slice)
