import glob
import os
import shutil


root = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/"
img_paths = sorted(glob.glob(root + '*.tif'))

# os.mkdir(root + "A172")
# os.mkdir(root + "BT474")
# os.mkdir(root + "BV2")
# os.mkdir(root + "Huh7")
# os.mkdir(root + "MCF7")
# os.mkdir(root + "SHSY5Y")
# os.mkdir(root + "SkBr3")
# os.mkdir(root + "SKOV3")

for img_path in img_paths:
    img_name = img_path.split('/')[-1]
    folder_name = img_name.split('_')[0]
    shutil.copyfile(img_path, root + folder_name + '/' + img_name)