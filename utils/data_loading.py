import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


def pad_mask(mask, repeatX, repeatY):
    """
    Relabels reflected instances in a segmentation mask after padding
    to ensure each instance has a unique label.

    Args:
        mask (torch.Tensor): Instance segmentation mask (H, W) with unique instance IDs.
        padding (int or tuple): Padding size.

    Returns:
        torch.Tensor: Mask with unique labels in the padded regions.
    """
    sizeX, sizeY = mask.size[0], mask.size[1]
    if mask.mode == "RGB":
        rgb = True
    else:
        rgb = False
    big_mask = torch.zeros((sizeY * repeatY, sizeX * repeatX), dtype=torch.int32)
    mask = np.array(mask)
    if rgb:
        mask = mask[:, :, 0]
    mask = mask.astype(np.float32)
    mask = torch.tensor(mask)
    max_label = torch.max(mask)
    if rgb:
        max_label = 0
    for i in range(repeatX):
        for j in range(repeatY):
            big_mask[j * sizeY : (j + 1) * sizeY, i * sizeX : (i + 1) * sizeX] = mask
            mask[mask > 0] = mask[mask > 0] + max_label
    if rgb:
        return transforms.functional.to_pil_image(big_mask, mode="I").convert("RGB")
    else:
        return transforms.functional.to_pil_image(big_mask)


def tiff_force_8bit(image, **kwargs):
    if image.format == "TIFF" and image.mode == "I;16":
        array = np.array(image)
        normalized = (
            (array.astype(np.uint16) - array.min())
            * 255.0
            / (array.max() - array.min())
        )
        image = Image.fromarray(normalized.astype(np.uint8))
    return image


def load_img(
    path,
    masks_path="",
    crop=512,
    loadMask=False,
    zoom=0,
    source="cellIm",
    fromMask=False,
):

    mask = None
    image = Image.open(path)
    image = tiff_force_8bit(image)
    image = image.convert("RGB")
    # Crop differently if the cell sizes should be matched betw the style and source
    x, y = crop, crop
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    if zoom != 0:
        image = image.resize(
            (int(image.size[0] * zoom), int(image.size[1] * zoom)),
            resample=Image.Resampling.LANCZOS,
        )
        # zoom out and pad if the style cells are smaller
        if zoom < 1:
            image = pad_mask(
                image, crop // image.size[0] + 1, crop // image.size[1] + 1
            )
    i, j, _, _ = transforms.RandomCrop(size=(crop, crop), padding=None).get_params(
        image, output_size=(crop, crop)
    )
    image = transforms.functional.crop(image, i, j, crop, crop)
    if loadMask:
        if source == "cellIm":
            mask_pt = masks_path + path.split("/")[-1][:-6] + "_GT_01.tif"
            mask = Image.open(mask_pt)
            mask = mask.resize(
                (mask.size[0] * 2, mask.size[1] * 2), resample=Image.Resampling.NEAREST
            )
            if zoom != 0:
                mask = mask.resize(
                    (int(mask.size[0] * zoom), int(mask.size[1] * zoom)),
                    resample=Image.Resampling.NEAREST,
                )
                if zoom < 1:
                    mask = pad_mask(
                        mask, crop // mask.size[0] + 1, crop // mask.size[1] + 1
                    )
            mask = transforms.functional.crop(mask, i, j, crop, crop)
        elif source == "liveCell":
            if fromMask:
                mask_pt = masks_path + path.split("/")[-1]
            else:
                mask_pt = masks_path + "mask_" + path.split("/")[-1]
            try:
                mask = Image.open(mask_pt)
            except:
                return None, None, True
            if zoom != 0:
                mask = mask.resize(
                    (int(mask.size[0] * zoom), int(mask.size[1] * zoom)),
                    resample=Image.Resampling.NEAREST,
                )
                if zoom < 1:
                    mask = pad_mask(
                        mask, crop // mask.size[0] + 1, crop // mask.size[1] + 1
                    )
            mask = transforms.functional.crop(mask, i, j, crop, crop)
        elif source == "CTC":
            if fromMask:
                mask_pt = path
            else:
                mask_pt = masks_path + "man_seg" + path.split("/")[-1][1:]
            mask = Image.open(mask_pt)
            if zoom != 0:
                mask = mask.resize(
                    (int(mask.size[0] * zoom), int(mask.size[1] * zoom)),
                    resample=Image.Resampling.NEAREST,
                )
                if zoom < 1:
                    mask = pad_mask(
                        mask, crop // mask.size[0] + 1, crop // mask.size[1] + 1
                    )
            mask = transforms.functional.crop(mask, i, j, crop, crop)
        # Skip the content images that have no cells
        if np.all(np.array(mask) == 0):
            return None, None, True
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0, mask, False
