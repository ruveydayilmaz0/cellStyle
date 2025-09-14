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
import matplotlib.pyplot as plt
from pathlib import Path
from utils.data_loading import load_img

feat_maps = []


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
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        ori_keys = sty_feat.keys()

        for ori_key in ori_keys:
            if ori_key[-1] == "q":
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key[-1] == "k" or ori_key[-1] == "v":
                feat_maps[i][ori_key] = sty_feat[ori_key]
    return feat_maps


def std_attn(feat_maps):
    stds = []
    for feat_map in feat_maps:
        std_ts = []
        sum = torch.Tensor([0]).to("cuda")
        for key in feat_map.keys():
            if key[-1] == "q":
                q = feat_map[key]
            elif key[-1] == "k":
                k = feat_map[key]
                attn = torch.einsum("b i d, b j d -> b i j", q, k)
                sum += attn.std(dim=-1).mean()
        stds.append((sum / 6).detach().cpu().numpy())
    return stds


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
        "--config",
        type=str,
        default="configs/config.sh",
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
    sty_img_list = sorted(os.listdir(opt.sty))
    cnt_img_list = sorted(os.listdir(opt.cnt))  # [91:]#[1100:]
    cnt_img_list = [file for file in cnt_img_list if "w2" not in file]

    # count for img name
    begin = time.time()
    before = True
    # pick a few imgs to calculate the stds
    for sty_name in np.random.choice(sty_img_list, size=3):
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

        for cnt_name in np.random.choice(cnt_img_list, size=3):
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
                feat_path_root, os.path.basename(cnt_name).split(".")[0] + "_cnt.pkl"
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

                        # Calculate the std of attention maps
                        if before:
                            map_cnt = np.array(std_attn(cnt_feat)) / 9
                            map_sty = np.array(std_attn(sty_feat)) / 9
                            map_feats = np.array(std_attn(feat_maps)) / 9
                            before = False
                        else:
                            map_cnt += np.array(std_attn(cnt_feat)) / 9
                            map_sty += np.array(std_attn(sty_feat)) / 9
                            map_feats += np.array(std_attn(feat_maps)) / 9

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))

    axs[0].plot(range(1, 51), map_cnt)
    axs[0].set_xlabel("Diffusion Timestep")
    axs[0].set_ylabel("Average Std")
    axs[0].set_title("Content Feature Maps")

    # maps = std_attn(sty_feat)
    axs[1].plot(range(1, 51), map_sty)
    axs[1].set_xlabel("Diffusion Timestep")
    axs[1].set_ylabel("Average Std")
    axs[1].set_title("Style Feature Maps")

    axs[2].plot(
        range(1, 51),
        (map_cnt + map_sty) / 2,
        linestyle="dashdot",
        color="r",
        label="Avr Content & Style",
    )
    axs[2].plot(
        range(1, 51), map_feats, linestyle="--", color="b", label="Injected Features"
    )
    axs[2].legend()
    axs[2].set_xlabel("Diffusion Timestep")
    axs[2].set_ylabel("Average Std")

    ratio = np.divide((map_cnt + map_sty) / 2, map_feats)
    axs[3].plot(range(1, 51), ratio, linestyle="dashdot", color="r", label="Std Ratio")
    axs[3].plot(
        range(1, 51),
        [np.average(ratio)] * 50,
        linestyle="--",
        color="g",
        label="Avr Ratio",
    )
    axs[3].legend()
    axs[3].set_xlabel("Diffusion Timestep")
    axs[3].set_ylabel("Std Ratio")

    plt.tight_layout()
    pair_name = opt.output_path.split("/")[-2]
    Path("std_plots/" + pair_name).mkdir(parents=True, exist_ok=True)
    plt.savefig("std_plots/" + pair_name + "/" + "avr_stds.png")

    print(f"Total end: {time.time() - begin}")


if __name__ == "__main__":
    main()
