from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from shapely.geometry import Polygon

# %matplotlib inline

coco = COCO('/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/train_huh7.json')

# for img_id in coco.imgs:
#     img = coco.imgs[img_id]
#     cat_ids = coco.getCatIds()
#     anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
#     anns = coco.loadAnns(anns_ids)
#     # coco.showAnns(anns)
#     mask = coco.annToMask(anns[0]).astype(np.uint16)
#     for i, ann in enumerate(anns):
#         mask += coco.annToMask(ann)*i
#     io.imsave("/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/GTmasks_a172/instance/mask_"+coco.imgs[img_id]['file_name'], mask)
#     mask = mask.astype(np.uint8)
#     mask[mask>0] = 255
#     io.imsave("/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/GTmasks_a172/semantic/mask_"+coco.imgs[img_id]['file_name'], mask)

def calculate_area(segmentation):
    """Calculate the area of a polygon from segmentation."""
    if isinstance(segmentation, list):  # Polygon format
        poly = Polygon([(segmentation[0][i], segmentation[0][i + 1]) for i in range(0, len(segmentation[0]), 2)])
        return poly.area
    return 0

# Loop through images
for img_id in coco.imgs:
    img = coco.imgs[img_id]
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)

    # Calculate areas for all annotations
    for ann in anns:
        ann['area'] = calculate_area(ann['segmentation'])

    # Sort annotations by area (ascending or descending)
    anns = sorted(anns, key=lambda x: x['area'], reverse=True)  # Smaller areas will overwrite larger ones

    # Create an empty mask for the instance segmentation map
    mask = np.zeros((img['height'], img['width']), dtype=np.uint16)

    # Add annotations to the mask
    for i, ann in enumerate(anns, start=1):  # Start indexing at 1
        ann_mask = coco.annToMask(ann)
        mask[ann_mask > 0] = i  # Assign the annotation index to the mask
    
    io.imsave("/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/GTmasks_huh7/instance/mask_"+coco.imgs[img_id]['file_name'], mask)
    mask = mask.astype(np.uint8)
    mask[mask>0] = 255
    io.imsave("/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/GTmasks_huh7/semantic/mask_"+coco.imgs[img_id]['file_name'], mask)
##############
# coco = COCO('/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/train_bv2.json')
# img_dir = '/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/'
# image_id = 817557

# img = coco.imgs[image_id]

# image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
# # plt.imshow(image, interpolation='nearest')

# plt.imshow(image)
# cat_ids = coco.getCatIds()
# anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
# anns = coco.loadAnns(anns_ids)
# # coco.showAnns(anns)
# mask = coco.annToMask(anns[0])
# for i, ann in enumerate(anns):
#     mask += coco.annToMask(ann)*i
# io.imsave('livecellmask.tif', mask)
# plt.imshow(mask)
# plt.show()
##############
# import json
# import cv2
# import numpy as np

# def create_binary_mask(image_id, annotations):
#     # Find the annotations for the given image_id
#     image_annotations = [annotation for annotation in annotations if annotation['image_id'] == image_id]

#     # Create an empty mask for the image
#     semantic_mask = np.zeros((520, 704), dtype=np.uint16)
#     instance_mask = np.zeros((520, 704), dtype=np.uint16)

#     # Iterate over annotations and draw polygons on the mask
#     for i,annotation in enumerate(image_annotations):
#         segmentation = np.array(annotation['segmentation']).reshape((-1, 2)).astype(np.int32)
#         # do this when creating input for style transfer
#         # cv2.fillPoly(semantic_mask, [segmentation], color=255)
#         cv2.polylines(semantic_mask, [segmentation], isClosed=True, color=0, thickness=1)
#         # cv2.fillPoly(instance_mask, [segmentation], color=i+1)
#         cv2.polylines(instance_mask, [segmentation], isClosed=True, color=0, thickness=1)

#     return semantic_mask, instance_mask

  
# json_file_path = '/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/train_bv2.json'

# with open(json_file_path, 'r') as json_file:
#     data = json.load(json_file)

#     # Iterate over images and create binary masks
#     for image_info in data['images']:
#         image_id = image_info['id']
#         filename = image_info['file_name']
#         semantic_mask, instance_mask = create_binary_mask(image_id, data['annotations'])

#         # Save the mask to a file or use it as needed
#         mask_filename = f"mask_{filename}"
#         cv2.imwrite("/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/GTmasks_bv2/semantic/" + mask_filename, semantic_mask)
#         cv2.imwrite("/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/GTmasks_bv2/instance/" + mask_filename, instance_mask)
#         print(f"Binary mask saved for image {image_id} as {mask_filename}")
