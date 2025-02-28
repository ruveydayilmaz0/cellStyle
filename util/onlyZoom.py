import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import pickle
from skimage import io
from pathlib import Path

feat_maps = []


def save_img_from_sample(model, samples_ddim, fname):
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255.0 * rearrange(x_image_torch[0].cpu().numpy(), "c h w -> h w c")
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(fname)


def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    feat_maps = [
        {
            "config": {
                "gamma": opt.gamma,
                "T": opt.T,
                "timestep": _,
            }
        }
        for _ in range(50)
    ]

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        ori_keys = sty_feat.keys()

        for ori_key in ori_keys:
            if ori_key[-1] == "q":
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key[-1] == "k" or ori_key[-1] == "v":
                feat_maps[i][ori_key] = sty_feat[ori_key]
    return feat_maps


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
        # big_mask = torch.zeros((sizeX * repeatX, sizeY * repeatY, 3), dtype=torch.int32)
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
            # max_label += max_init
            # if rgb:
            #     big_mask[i*sizeX:(i+1)*sizeX, j*sizeY:(j+1)*sizeY, :] = mask
            # else:
            big_mask[j * sizeY : (j + 1) * sizeY, i * sizeX : (i + 1) * sizeX] = mask
            mask[mask > 0] = mask[mask > 0] + max_label
    if rgb:
        # return transforms.functional.to_pil_image(big_mask.permute(2,0,1), mode="RGB")
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
        # if the cell borders are too dark, make them brighter
        # normalized[normalized>80] = 80
        # normalized = 3 * normalized
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
    # Downsize the U2OS images to match the cell sizes to PhC cells
    if zoom != 0:
        image = image.resize(
            (int(image.size[0] * zoom), int(image.size[1] * zoom)),
            resample=Image.Resampling.LANCZOS,
        )
        # zoom out and pad if the style cells are smaller
        if zoom < 1:
            # padding = (crop // 2, crop // 2, crop // 2, crop // 2)
            # image = transforms.functional.pad(image, padding, padding_mode='reflect')
            # repeat instead of mirroring
            image = pad_mask(
                image, crop // image.size[0] + 1, crop // image.size[1] + 1
            )
    i, j, _, _ = transforms.RandomCrop(size=(crop, crop), padding=None).get_params(
        image, output_size=(crop, crop)
    )
    image = transforms.functional.crop(image, i, j, crop, crop)
    if loadMask:
        if source == "cellIm":
            # masks_path = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cell_im_lib_MP6843/MP6843_segInstance/"
            mask_pt = masks_path + path.split("/")[-1][:-6] + "_GT_01.tif"
            mask = Image.open(mask_pt)
            # In the MP6843 dataset, masks are half the size of images, reverse this first
            mask = mask.resize(
                (mask.size[0] * 2, mask.size[1] * 2), resample=Image.Resampling.NEAREST
            )
            if zoom != 0:
                mask = mask.resize(
                    (int(mask.size[0] * zoom), int(mask.size[1] * zoom)),
                    resample=Image.Resampling.NEAREST,
                )
                if zoom < 1:
                    # mask = transforms.functional.pad(mask, padding, padding_mode='constant', fill=0)
                    mask = pad_mask(
                        mask, crop // mask.size[0] + 1, crop // mask.size[1] + 1
                    )
            mask = transforms.functional.crop(mask, i, j, crop, crop)
            # mask = mask.resize((reshape, reshape), resample=Image.Resampling.NEAREST)
        elif source == "liveCell":
            # masks_path = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/GTmasks_shsy5y/instance/"
            if fromMask:
                mask_pt = masks_path + path.split("/")[-1]
            else:
                mask_pt = masks_path + "mask_" + path.split("/")[-1]
            # some masks are missing in this dataset if so, return
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
                    # mask = transforms.functional.pad(mask, padding, padding_mode='constant', fill=0)
                    mask = pad_mask(
                        mask, crop // mask.size[0] + 1, crop // mask.size[1] + 1
                    )
            mask = transforms.functional.crop(mask, i, j, crop, crop)
        elif source == "CTC":
            # masks_path = "/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2020/PhC-C2DL-PSC/01_ST/SEG/"
            if fromMask:
                mask_pt = masks_path + "t" + path.split("/")[-1][1:]
            else:
                mask_pt = masks_path + "man_seg" + path.split("/")[-1][1:]
            mask = Image.open(mask_pt)
            if zoom != 0:
                mask = mask.resize(
                    (int(mask.size[0] * zoom), int(mask.size[1] * zoom)),
                    resample=Image.Resampling.NEAREST,
                )
                if zoom < 1:
                    # mask = transforms.functional.pad(mask, padding, padding_mode='constant', fill=0)
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


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/work/scratch/yilmaz/StyleID/configs/onlyZoom/liveCell_shsy5y_mp6843.sh",
        help="path to the config file",
    )
    args = parser.parse_args()

    # Load the config file
    config = OmegaConf.load(args.config)

    # Set the arguments from the config file
    opt = argparse.Namespace(**config)
    feat_path_root = opt.precomputed

    seed_everything(22)
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)
    if len(feat_path_root) > 0:
        os.makedirs(feat_path_root, exist_ok=True)

    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    seed = torch.initial_seed()
    opt.seed = seed

    global feat_maps
    feat_maps = [{"config": {"gamma": opt.gamma, "T": opt.T}} for _ in range(50)]

    cnt_img_list = sorted(os.listdir(opt.cnt))  # [1100:]
    cnt_img_list = [file for file in cnt_img_list if "w2" not in file]

    # count for img name
    m = 0
    begin = time.time()

    for cnt_name in cnt_img_list:
        cnt_name_ = os.path.join(opt.cnt, cnt_name)
        init_cnt, mask, no_mask = load_img(
            cnt_name_,
            masks_path=opt.masks_path,
            crop=512,
            loadMask=True,
            zoom=opt.cnt_zoom,
            source=opt.source_type,
            fromMask=opt.fromMask,
        )
        # if no mask present for the source image, skip this iteration
        if no_mask:
            continue

        # Save in the CTC format
        Path(output_path + "01").mkdir(parents=True, exist_ok=True)
        Path(output_path + "01_ST/SEG").mkdir(parents=True, exist_ok=True)
        # init_cnt.convert('L').save()
        # init_cnt = init_cnt.cpu().permute(0, 2, 3, 1).numpy()
        # init_cnt = torch.from_numpy(init_cnt).permute(0, 3, 1, 2)
        init_cnt = torch.clamp((init_cnt[0] + 1.0) / 2.0, min=0.0, max=1.0)
        init_cnt = 255.0 * rearrange(init_cnt.cpu().numpy(), "c h w -> h w c")
        init_cnt = Image.fromarray(init_cnt.astype(np.uint8))
        init_cnt.convert("L").save(output_path + "01/t" + str(m).zfill(4) + ".tif")
        mask.save(output_path + "01_ST/SEG/man_seg" + str(m).zfill(4) + ".tif")
        m = m + 1

    print(f"Total end: {time.time() - begin}")


if __name__ == "__main__":
    main()
