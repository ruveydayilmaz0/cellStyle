# Code adapted from https://github.com/jiwoogit/StyleID

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
import time
import pickle
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


def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)
    output = ((cnt_feat - cnt_mean) / cnt_std) * sty_std + sty_mean
    return output


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
        "--config", type=str, default="configs/try.sh", help="path to the config file"
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

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(",")))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(
        ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False
    )
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed

    global feat_maps
    feat_maps = [{"config": {"gamma": opt.gamma, "T": opt.T}} for _ in range(50)]

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, "z_enc", i)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1

    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks, i, "output_block")

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    start_step = opt.start_step
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    sty_img_list = sorted(
        os.listdir(opt.sty)
    )  # [1100:]#for musc only, dont look at the initial files which are almost empty
    cnt_img_list = sorted(os.listdir(opt.cnt))  # [91:]#[1100:]
    cnt_img_list = [file for file in cnt_img_list if "w2" not in file]

    # count for img name
    m = 0
    begin = time.time()
    # define a txt file
    with open(output_path + "filenames.txt", "w") as file:
        file.write("Style\tContent\tOutput\n")
        for sty_name in sty_img_list:
            sty_name_ = os.path.join(opt.sty, sty_name)
            init_sty, _, _ = load_img(sty_name_, crop=opt.H)
            init_sty = init_sty.to(device)
            seed = -1
            sty_feat_name = os.path.join(
                feat_path_root, os.path.basename(sty_name).split(".")[0] + "_sty.pkl"
            )
            sty_z_enc = None

            if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
                print("Precomputed style feature loading: ", sty_feat_name)
                with open(sty_feat_name, "rb") as h:
                    sty_feat = pickle.load(h)
                    sty_z_enc = torch.clone(sty_feat[0]["z_enc"])
            else:
                init_sty = model.get_first_stage_encoding(
                    model.encode_first_stage(init_sty)
                )
                sty_z_enc, _ = sampler.encode_ddim(
                    init_sty.clone(),
                    num_steps=ddim_inversion_steps,
                    unconditional_conditioning=uc,
                    end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
                    callback_ddim_timesteps=save_feature_timesteps,
                    img_callback=ddim_sampler_callback,
                )
                sty_feat = copy.deepcopy(feat_maps)
                sty_z_enc = feat_maps[0]["z_enc"]

            for cnt_name in cnt_img_list:
                cnt_name_ = os.path.join(opt.cnt, cnt_name)
                init_cnt, mask, no_mask = load_img(
                    cnt_name_,
                    masks_path=opt.masks_path,
                    crop=opt.H,
                    loadMask=True,
                    zoom=opt.cnt_zoom,
                    source=opt.source_type,
                    fromMask=opt.fromMask,
                )
                # if no mask present for the source image, skip this iteration
                if no_mask:
                    continue
                init_cnt = init_cnt.to(device)
                cnt_feat_name = os.path.join(
                    feat_path_root,
                    os.path.basename(cnt_name).split(".")[0] + "_cnt.pkl",
                )
                cnt_feat = None

                # ddim inversion encoding
                if len(feat_path_root) > 0 and os.path.isfile(cnt_feat_name):
                    print("Precomputed content feature loading: ", cnt_feat_name)
                    with open(cnt_feat_name, "rb") as h:
                        cnt_feat = pickle.load(h)
                        cnt_z_enc = torch.clone(cnt_feat[0]["z_enc"])
                else:
                    init_cnt = model.get_first_stage_encoding(
                        model.encode_first_stage(init_cnt)
                    )
                    cnt_z_enc, _ = sampler.encode_ddim(
                        init_cnt.clone(),
                        num_steps=ddim_inversion_steps,
                        unconditional_conditioning=uc,
                        end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
                        callback_ddim_timesteps=save_feature_timesteps,
                        img_callback=ddim_sampler_callback,
                    )
                    cnt_feat = copy.deepcopy(feat_maps)
                    cnt_z_enc = feat_maps[0]["z_enc"]

                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            # inversion
                            print(f"Inversion end: {time.time() - begin}")
                            if opt.without_init_adain:
                                adain_z_enc = cnt_z_enc
                            else:
                                adain_z_enc = adain(cnt_z_enc, sty_z_enc)
                            feat_maps = feat_merge(
                                opt, cnt_feat, sty_feat, start_step=start_step
                            )
                            if opt.without_attn_injection:
                                feat_maps = None

                            # inference
                            samples_ddim, intermediates = sampler.sample(
                                S=ddim_steps,
                                batch_size=1,
                                shape=shape,
                                verbose=False,
                                unconditional_conditioning=uc,
                                eta=opt.ddim_eta,
                                x_T=adain_z_enc,
                                injected_features=feat_maps,
                                start_step=start_step,
                            )

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                            )
                            x_samples_ddim = (
                                x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                            )
                            x_image_torch = torch.from_numpy(x_samples_ddim).permute(
                                0, 3, 1, 2
                            )
                            x_sample = 255.0 * rearrange(
                                x_image_torch[0].cpu().numpy(), "c h w -> h w c"
                            )
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            # Save in the CTC format
                            Path(output_path + "01").mkdir(parents=True, exist_ok=True)
                            Path(output_path + "01_ST/SEG").mkdir(
                                parents=True, exist_ok=True
                            )
                            if opt.rgb:
                                img.save(
                                    output_path + "01/t" + str(m).zfill(4) + ".tif"
                                )
                            else:
                                img.convert("L").save(
                                    output_path + "01/t" + str(m).zfill(4) + ".tif"
                                )
                            mask.save(
                                output_path
                                + "01_ST/SEG/man_seg"
                                + str(m).zfill(4)
                                + ".tif"
                            )
                            # Save the names in txt file
                            out_name = str(m).zfill(4) + ".tif"
                            file.write(f"{sty_name}\t{cnt_name}\t{out_name}\n")
                            m = m + 1
                            if len(feat_path_root) > 0:
                                print("Save features")
                                if not os.path.isfile(cnt_feat_name):
                                    with open(cnt_feat_name, "wb") as h:
                                        pickle.dump(cnt_feat, h)
                                if not os.path.isfile(sty_feat_name):
                                    with open(sty_feat_name, "wb") as h:
                                        pickle.dump(sty_feat, h)
                            # stop if 4000 images are generated
                            if m >= 4000:
                                break
            else:
                continue  # Only runs if inner loop didn’t break
            break

    print(f"Total end: {time.time() - begin}")


if __name__ == "__main__":
    main()
