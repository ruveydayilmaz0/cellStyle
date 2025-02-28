import os
from pathlib import Path
import shutil
from PIL import Image
import glob

# ## MP6843 ##
# source = '/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/nuinsseg/archive/human placenta/tissue images/'
# destination = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/nuinsseg/archive/CTC_format_human_placenta/"

# Path(destination+'01/').mkdir(parents=True, exist_ok=True)
# Path(destination+'01_ST/SEG').mkdir(parents=True, exist_ok=True)

# img_list = sorted(glob.glob(source + '*.png'))
# # img_list = [file for file in img_list if 'w2' not in file]
# masks_path = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/nuinsseg/archive/human placenta/label masks modify/"
# i = 0
# for img_pt in img_list:
#     # img_pt = source+'MP6843_img_full/'+img_pt
#     # mask_pt = masks_path + 'mask_' + img_pt.split('/')[-1]
#     mask_pt = masks_path + img_pt.split('/')[-1].replace('.png', '.tif')
#     if os.path.exists(mask_pt):
#         # shutil.copy(img_pt, destination+"01/t"+str(i).zfill(3)+'.tif')
#         # need to double the mask size
#         img = Image.open(img_pt).convert("RGB")
#         img.save(destination+"01/t"+str(i).zfill(3)+'.tif')
#         # mask = Image.open(mask_pt)
#         # mask = mask.resize((mask.size[0]*2, mask.size[1]*2), resample=Image.Resampling.NEAREST)
#         # mask.save(destination+"01_ST/SEG/man_seg"+str(i).zfill(3)+'.tif')
#         shutil.copy(mask_pt, destination+"01_ST/SEG/man_seg"+str(i).zfill(3)+'.tif')
#         i = i+1

# ## Livecell ##
# source = '/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/'
# mask_folder = "GTmasks_skov3/instance/"
# img_folder = "images/livecell_train_val_images/SKOV3/"
# destination = source + img_folder + "CTC_format/"

# Path(destination+'01/').mkdir(parents=True, exist_ok=True)
# Path(destination+'01_ST/SEG').mkdir(parents=True, exist_ok=True)

# img_list = sorted(glob.glob(source + img_folder + '*.tif'))

# for i, img_pt in enumerate(img_list):
#     mask_pt = source + mask_folder + "mask_" + img_pt.split('/')[-1]
#     if os.path.exists(mask_pt):
#         shutil.copy(img_pt, destination+"01/t"+str(i).zfill(4)+'.tif')
#         shutil.copy(mask_pt, destination+"01_ST/SEG/man_seg"+str(i).zfill(4)+'.tif')

## Omnipose ##
sources = [
    "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/omnipose/datasets/bact_fluor/",
    "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/omnipose/datasets/bact_phase/",
]
destination = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/omnipose/datasets/CTC_format/"

Path(destination + "01/").mkdir(parents=True, exist_ok=True)
Path(destination + "01_ST/SEG").mkdir(parents=True, exist_ok=True)
i = 0
for source in sources:
    sub_folder_list = sorted(glob.glob(source + "train_sorted/*"))
    sub_folder_list.extend(sorted(glob.glob(source + "test_sorted/*")))
    for sub_folder in sub_folder_list:
        img_list = sorted(glob.glob(sub_folder + "/*.tif"))
        for img_pt in img_list:
            if "mask" not in img_pt and "flows" not in img_pt:
                mask_pt = img_pt.replace(".tif", "") + "_masks.tif"
                if os.path.exists(mask_pt):
                    shutil.copy(img_pt, destination + "01/t" + str(i).zfill(4) + ".tif")
                    shutil.copy(
                        mask_pt,
                        destination + "01_ST/SEG/man_seg" + str(i).zfill(4) + ".tif",
                    )
                    i = i + 1
