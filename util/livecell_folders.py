import glob
import os
import shutil
from skimage import io
import numpy as np

root = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/GTmasks_shsy5y/instance/"
destination = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/SHSY5Y_lessCrowded/"
img_paths = sorted(glob.glob(root + "*.tif"))

# os.mkdir(root + "A172")
# os.mkdir(root + "BT474")
# os.mkdir(root + "BV2")
# os.mkdir(root + "Huh7")
# os.mkdir(root + "MCF7")
# os.mkdir(root + "SHSY5Y")
# os.mkdir(root + "SkBr3")
# os.mkdir(root + "SKOV3")

for img_path in img_paths:
    img = io.imread(img_path)
    max = np.max(img)
    img[img > max // 3] = 0
    io.imsave(destination + img_path.split("/")[-1], img)
    # img_name = img_path.split('/')[-1]
    # folder_name = img_name.split('_')[0]
    # shutil.copyfile(img_path, root + 'man_track' + img_name[-7:])
