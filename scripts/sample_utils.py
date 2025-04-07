import os
import yaml
import numpy as np
from PIL import Image

import torch

from lvdm.utils.common_utils import instantiate_from_config
from lvdm.utils.saving_utils import npz_to_video_grid, npz_to_imgsheet_5d

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    if sample.dim() == 5:
        sample = sample.permute(0, 2, 3, 4, 1)
    else:
        sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample

def save_args(save_dir, args):
    fpath = os.path.join(save_dir, "sampling_args.yaml")
    with open(fpath, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

# ------------------------------------------------------------------------------------------
def load_model(config, ckpt_path, gpu_id=None):
    print(f"Loading model from {ckpt_path}")
    
    # load sd
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    try:
        global_step = pl_sd["global_step"]
        epoch = pl_sd["epoch"]
    except:
        global_step = -1
        epoch = -1
    
    # load sd to model
    try:
        sd = pl_sd["state_dict"]
    except:
        sd = pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=True)

    # move to device & evalconnection
    if gpu_id is not None:
        model.to(f"cuda:{gpu_id}")
    else:
        model.cuda()
    model.eval()

    return model, global_step, epoch

def make_sample_dir(opt, global_step, epoch):
    if not getattr(opt, 'not_automatic_logdir', False):
        gs_str = f"globalstep{global_step:09}" if global_step is not None else "None"
        e_str = f"epoch{epoch:06}" if epoch is not None else "None"
        ckpt_dir = os.path.join(opt.logdir, f"{gs_str}_{e_str}")
        
        # subdir name
        if opt.prompt_file is not None:
            subdir = f"prompts_{os.path.splitext(os.path.basename(opt.prompt_file))[0]}"
        else:
            subdir = f"prompt_{opt.prompt[:10]}"
        subdir += "_DDPM" if opt.vanilla_sample else f"_DDIM{opt.custom_steps}steps"
        subdir += f"_CfgScale{opt.scale}"
        if opt.cond_fps is not None:
            subdir += f"_fps{opt.cond_fps}"
        if opt.seed is not None:
            subdir += f"_seed{opt.seed}"

        return os.path.join(ckpt_dir, subdir)
    else:
        return opt.logdir

# ------------------------------------------------------------------------------------------
@torch.no_grad()
def get_conditions(prompts, model, batch_size, cond_fps=None,):
    
    if isinstance(prompts, str) or isinstance(prompts, int):
        prompts = [prompts]
    if isinstance(prompts, list):
        if len(prompts) == 1:
            prompts = prompts * batch_size
        elif len(prompts) == batch_size:
            pass
        else:
            raise ValueError(f"invalid prompts length: {len(prompts)}")
    else:
        raise ValueError(f"invalid prompts: {prompts}")
    assert(len(prompts) == batch_size)
    
    # content condition: text / class label
    c = model.get_learned_conditioning(prompts)
    key = 'c_concat' if model.conditioning_key == 'concat' else 'c_crossattn'
    c = {key: [c]}

    # temporal condition: fps
    if getattr(model, 'cond_stage2_config', None) is not None:
        if model.cond_stage2_key == "temporal_context":
            assert(cond_fps is not None)
            batch = {'fps': torch.tensor([cond_fps] * batch_size).long().to(model.device)}
            fps_embd = model.cond_stage2_model(batch)
            c[model.cond_stage2_key] = fps_embd
    
    return c

# ------------------------------------------------------------------------------------------
@torch.no_grad()
def get_multi_conditions(prompts, model, batch_size, cond_fps=None,):
    m_c = []
    for pt in prompts:
        # content condition: text / class label
        c = model.get_learned_conditioning(pt)
        key = 'c_concat' if model.conditioning_key == 'concat' else 'c_crossattn'
        c = {key: [c]}

        # temporal condition: fps
        if getattr(model, 'cond_stage2_config', None) is not None:
            if model.cond_stage2_key == "temporal_context":
                assert(cond_fps is not None)
                batch = {'fps': torch.tensor([cond_fps] * batch_size).long().to(model.device)}
                fps_embd = model.cond_stage2_model(batch)
                c[model.cond_stage2_key] = fps_embd
        m_c.append(c)
    return m_c

# ------------------------------------------------------------------------------------------
def make_model_input_shape(model, batch_size, T=None):
    image_size = [model.image_size, model.image_size] if isinstance(model.image_size, int) else model.image_size
    C = model.model.diffusion_model.in_channels
    if T is None:
        T = model.model.diffusion_model.temporal_length
    shape = [batch_size, C, T, *image_size]
    return shape

# ------------------------------------------------------------------------------------------
def sample_batch(model, noise_shape, 
                 condition, 
                 uc=None,
                 sample_type="ddim",
                 sampler=None,
                 ddim_steps=None, 
                 eta=None,
                 unconditional_guidance_scale=1.0, 
                 denoising_progress=False,
                 timesteps=None,
                 #------------------------------------Edit part
                 prev_clip=None,
                 pred_x0=None,
                 lfai_scale=0.0,
                 sgs_scale=0.0,
                 **kwargs,
                 ):
    if sample_type == "ddpm":
        samples = model.p_sample_loop(cond=condition, shape=noise_shape,
                                      return_intermediates=False, 
                                      verbose=denoising_progress,
                                      )
    elif sample_type == "ddim":
        assert(sampler is not None)
        assert(ddim_steps is not None)
        assert(eta is not None)
        ddim_sampler = sampler
        sample_latent, sam_pred_x0 = ddim_sampler.sample(
                                                        S=ddim_steps,
                                                        batch_size=noise_shape[0],
                                                        shape=noise_shape[1:],
                                                        conditioning=condition,
                                                        eta=eta,
                                                        unconditional_conditioning=uc,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        timesteps=timesteps,
                                                        #------------------------------------Edit part
                                                        prev_clip=prev_clip,
                                                        pred_x0=pred_x0,
                                                        lfai_scale=lfai_scale,
                                                        sgs_scale=sgs_scale,
                                                        **kwargs,
                                                        )
    else:
        raise ValueError
    return sample_latent, sam_pred_x0

# ------------------------------------------------------------------------------------------
def torch_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    if sample.dim() == 5:
        sample = sample.permute(0, 2, 3, 4, 1)
    else:
        sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample
# ------------------------------------------------------------------------------------------
def save_results(videos, save_dir, 
                 save_name="results", save_fps=8, save_mp4=True, 
                 save_npz=False, save_mp4_sheet=False, save_jpg=False
                 ):
    if save_mp4:
        save_subdir = os.path.join(save_dir, "videos")
        # os.makedirs(save_subdir, exist_ok=True)
        shape_str = "x".join([str(x) for x in videos[0:1,...].shape])
        for i in range(videos.shape[0]):
            npz_to_video_grid(videos[i:i+1,...], 
                              os.path.join(save_dir, f"{save_name}_{i:03d}_{shape_str}.mp4"), #save_subdir -> save_dir
                              fps=save_fps)
        print(f'Successfully saved videos in {save_subdir}')
    
    shape_str = "x".join([str(x) for x in videos.shape])
    if save_npz:
        save_path = os.path.join(save_dir, f"{save_name}_{shape_str}.npz")
        np.savez(save_path, videos)
        print(f'Successfully saved npz in {save_path}')
    
    if save_mp4_sheet:
        save_path = os.path.join(save_dir, f"{save_name}_{shape_str}.mp4")
        npz_to_video_grid(videos, save_path, fps=save_fps)
        print(f'Successfully saved mp4 sheet in {save_path}')

    if save_jpg:
        print(os.path.isdir(save_dir+"/jpg"))
        os.makedirs(save_dir+"/jpg", exist_ok=True)
        print(os.path.isdir(save_dir+"/jpg"))

        for frames in range(videos.shape[1]):
            # new_video = [np.expand_dims(videos[:,0], axis=0), np.expand_dims(videos[:,15], axis=0)]
            # videos = np.concatenate(new_video, 1)
            clip, frame = divmod(frames, 16)
            save_path = os.path.join(save_dir+"/jpg", f"{clip}_{frame}.jpg")
            npz_to_imgsheet_5d(np.expand_dims(videos[:,frames], axis=0), save_path, nrow=videos.shape[1])
        print(f'Successfully saved jpg sheet in {save_path}')