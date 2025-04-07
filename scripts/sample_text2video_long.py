import os
import sys

import time
import pandas as pd
import argparse
import math
from tqdm import trange
import torch
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from data.dataset import ImageSequenceDataset

from lvdm.samplers.ddim import DDIMSampler
from lvdm.utils.common_utils import str2bool
from scripts.sample_utils import (load_model, 
                                  get_conditions, get_multi_conditions, make_model_input_shape, torch_to_np, sample_batch, 
                                  save_results,
                                  save_args
                                  )
from data.image import img2frame



# ------------------------------------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser()
    # basic args
    parser.add_argument("--ckpt_path", type=str, help="model checkpoint path")
    parser.add_argument("--config_path", type=str, help="model config path (a yaml file)")
    parser.add_argument("--prompt", nargs='+', type=str, help="prompt csv file")
    parser.add_argument("--choice", nargs='+', type=int, help="choice prompt")
    parser.add_argument("--save_dir", type=str, help="results saving dir", default="results/")
    parser.add_argument("--path", type=str, help="upload img or frames", default="init/teaser_car")
    # sampling args
    parser.add_argument("--n_samples", type=int, help="how many samples for each text prompt", default=2)
    parser.add_argument("--batch_size", type=int, help="video batch size for sampling", default=1)
    parser.add_argument("--decode_frame_bs", type=int, help="frame batch size for framewise decoding", default=1)
    parser.add_argument("--sample_type", type=str, help="ddpm or ddim", default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, help="ddim sampling -- number of ddim denoising timesteps", default=50)
    parser.add_argument("--eta", type=float, help="ddim sampling -- eta (0.0 yields deterministic sampling, 1.0 yields random sampling)", default=1.0)
    parser.add_argument("--seed", type=int, default=None, help="fix a seed for randomness (If you want to reproduce the sample results)")
    parser.add_argument("--num_frames", type=int, default=16, help="number of input frames")
    parser.add_argument("--show_denoising_progress", action='store_true', default=False, help="whether show denoising progress during sampling one batch",)
    parser.add_argument("--cfg_scale", type=float, default=15.0, help="classifier-free guidance scale")
    parser.add_argument("--lfai_scale", type=float, default=1000.0, help="Last Frame-aware Latent Initialization scale")
    parser.add_argument("--sgs_scale", type=float, default=7.0, help="Structure-Guided Sampling scale")
    
    # saving args
    parser.add_argument("--save_mp4", type=str2bool, default=True, help="whether save samples in separate mp4 files", choices=["True", "true", "False", "false"])
    parser.add_argument("--save_mp4_sheet", action='store_true', default=False, help="whether save samples in mp4 file",)
    parser.add_argument("--save_npz", action='store_true', default=False, help="whether save samples in npz file",)
    parser.add_argument("--save_jpg", action='store_true', default=False, help="whether save samples in jpg file",)
    parser.add_argument("--save_fps", type=int, default=8, help="fps of saved mp4 videos",)
    return parser

# ------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_text2video(model, prompts, n_samples, batch_size, path,
                      sample_type="ddim", sampler=None, 
                      ddim_steps=50, eta=1.0, cfg_scale=7.5, 
                      decode_frame_bs=1,
                      show_denoising_progress=False,
                      #------------------------------------Edit part
                      num_frames=16,
                      lfai_scale=1000.0,
                      sgs_scale=7.0
                      ):
    
    # prompt(list) -> dict
    assert(model.cond_stage_model is not None)
    cond_embd = get_multi_conditions(prompts, model, batch_size)
    uncond_embd = get_conditions("", model, batch_size) if cfg_scale != 1.0 else None
    # video_dataset = img2frame(path).cuda()


    # sample batches
    all_videos = []
    n_iter = math.ceil(n_samples / batch_size)
    iterator  = trange(n_iter, desc="Sampling Batches (text-to-video)")
    for _ in iterator:
        noise_shape = make_model_input_shape(model, batch_size, T=num_frames)
        video_clip = []
        timesteps = None
        sample_latent = None
        sam_pred_x0 = None
        # start from X_T
        cond_embd.insert(0,None)

        # start from X_0
        for idx_p in range(len(prompts)):
            sample_latent, sam_pred_x0 = sample_batch(model, noise_shape, 
                                                    cond_embd[idx_p:],
                                                    uc=uncond_embd,
                                                    sample_type=sample_type, 
                                                    sampler=sampler,
                                                    ddim_steps=ddim_steps,
                                                    eta=eta,
                                                    unconditional_guidance_scale=cfg_scale,
                                                    denoising_progress=show_denoising_progress,
                                                    timesteps=timesteps,
                                                    #------------------------------------Edit part
                                                    prev_clip=sample_latent,
                                                    pred_x0=sam_pred_x0,
                                                    lfai_scale=lfai_scale,
                                                    sgs_scale=sgs_scale
                                                    )
            
            video_clip.append(sample_latent)

        full_video = torch.concatenate(video_clip, 2)
        samples = model.decode_first_stage(full_video, decode_bs=decode_frame_bs, return_cpu=False)
        all_videos.append(torch_to_np(samples))
        all_videos = np.concatenate(all_videos, axis=1)
        
    assert(all_videos.shape[0] >= n_samples)
    return all_videos
    
def register_recr(net_, count, place_in_unet):
        # print(net_)
        # print(net_.__class__.__name__, "\n")
        if net_.__class__.__name__ == 'CrossAttention':
            # print(net_.forward)
            return count + 1
        elif hasattr(net_, 'children'):
            for net in net_.named_children():
                if net !='attn_temporal':
                    count = register_recr(net[1], count, place_in_unet)
        return count
# ------------------------------------------------------------------------------------------
def main():
    """
    text-to-video generation
    """
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    os.makedirs(opt.save_dir, exist_ok=True)
    # save_args(opt.save_dir, opt) # save "sampling_args.yaml"

    # set seeds
    if opt.seed is not None:
        seed = opt.seed
        seed_everything(seed)

    # load & merge config
    config = OmegaConf.load(opt.config_path)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)
    print("config: \n", config)

    # get model & sampler
    model, _, _ = load_model(config, opt.ckpt_path)                
    ddim_sampler = DDIMSampler(model) if opt.sample_type == "ddim" else None

    # classify prompt type
    # read csv file or only prompts    
    if opt.prompt[0].endswith(".csv"):
        opt.prompt_file = opt.prompt[0]
        opt.prompt = None
    else:                
        opt.prompt_file = None

    start = time.time()
    for choice in opt.choice:
    # prepare prompt
        if opt.prompt_file is not None:
            data = pd.read_csv(opt.prompt_file, index_col=0)
            prompts = data.loc[choice].values
            prompts = [x for x in prompts if str(x) != 'nan']
        else:
            prompts = opt.prompt
    
        
        if opt.prompt_file is not None:
            save_seed_name = f"seed{seed}"
        else:
            prompts_str = ''                
            for prompt in prompts:
                prompts_str = prompts_str + prompt + '|'
            save_name = prompts_str.replace(" ", "_") if " " in prompts_str else prompts_str
            save_seed_name = save_name + f"seed{seed:05d}"

        print(f"Index: {choice}, Seed: {seed}!\n")
        samples = sample_text2video(model, prompts, opt.n_samples, opt.batch_size, opt.path,
                                    sample_type=opt.sample_type, sampler=ddim_sampler,
                                    ddim_steps=opt.ddim_steps, eta=opt.eta, cfg_scale=opt.cfg_scale,
                                    decode_frame_bs=opt.decode_frame_bs,
                                    show_denoising_progress=opt.show_denoising_progress,
                                    #------------------------------------Edit part
                                    num_frames=opt.num_frames,
                                    lfai_scale=opt.lfai_scale,
                                    sgs_scale=opt.sgs_scale
                                    )
            
            
        save_dir = opt.save_dir + f"IDX_{choice}"
        save_results(samples, save_dir, save_name=save_seed_name, save_fps=opt.save_fps)
        
    min, sec = divmod(time.time() - start, 60)
    print("Finish sampling!")
    print(f"Run time = {int(min)}\'{round(sec)}\"")

# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()