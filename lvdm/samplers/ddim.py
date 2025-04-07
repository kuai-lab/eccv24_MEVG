"""SAMPLING ONLY."""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from tqdm import tqdm
import clip
from PIL import Image
import torchvision.transforms as T
from lvdm.prompt_attention import attention_util
from lvdm.models.modules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

import datetime
import math


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               eta=0.,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               timesteps=None,
               #------------------------------------Edit part
               prev_clip=None,
               pred_x0=None,
               lfai_scale=0.0,
               sgs_scale=0.0
               ):

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
        
        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)
            
        sample_latent, sam_pred_x0 = self.ddim_sampling(conditioning, size,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=unconditional_conditioning,
                                                        timesteps=timesteps,
                                                        #------------------------------------Edit part
                                                        prev_clip=prev_clip,
                                                        pred_x0=pred_x0,
                                                        lfai_scale=lfai_scale,
                                                        sgs_scale=sgs_scale,
                                                        )
        return sample_latent, sam_pred_x0

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, timesteps=None, 
                      prev_clip=None, pred_x0=None, lfai_scale=0.0, sgs_scale=0.0):
        device = self.model.betas.device
        frames = shape[2]
        timesteps = self.ddim_timesteps
        total_steps = timesteps.shape[0]
        
        noise_latent = torch.randn(shape, device=device)
        sam_pred_x0=None

        sampler_iterator = tqdm(np.flip(timesteps), desc='DDIM Sampler', total=total_steps) 
        inversion_iterator = tqdm(timesteps, desc='DDIM Inversion', total=total_steps) 

        # Inversion
        if prev_clip is not None:
            """repeat-inversion(DN,LFAI)-sampling"""
            noise_latent = prev_clip[:,:,-1].unsqueeze(2).repeat(1,1,16,1,1)
            noise_latent= self.ddim_inversion(noise_latent, cond[1], unconditional_conditioning, inversion_iterator, device, step_pred=pred_x0, lfai_scale=lfai_scale)
            
        # Sampler
        curr_x0, sam_pred_x0 = self.ddim_noise2clean(noise_latent, cond[1], unconditional_conditioning, unconditional_guidance_scale, sampler_iterator, prev_clip, device, step_pred=pred_x0, sgs_scale=sgs_scale)
        return curr_x0, sam_pred_x0

    def ddim_noise2clean(self, noise_latent, cond_embd, uncond_embd, ucgs, iterator, prev_clip, device, step_pred=None, sgs_scale=0.0):
        curr_x0, pred_x0 = self.ddim_noise2clean_loop(noise_latent, cond_embd, uncond_embd, ucgs, iterator, prev_clip, device, step_pred, sgs_scale)
        return curr_x0, pred_x0
    
    @torch.no_grad()
    def ddim_noise2clean_loop(self, noise_latent, cond_embd, uncond_embd, ucgs, iterator, prev_clip, device, step_pred=None, sgs_scale=0.0):
        pred_x0=[]
        pred_guid=None
       

        for idx, timestep in enumerate(iterator):
            ts = torch.full((1,), timestep, device=device, dtype=torch.long)
            if step_pred is not None:
                pred_guid = step_pred[idx]
        
            eta=1
            e_t = self.model.apply_model(noise_latent, ts, cond_embd, prev_attn_map=None)
            e_t_uncond = self.model.apply_model(noise_latent, ts, uncond_embd)
            e_t = e_t_uncond + ucgs * (e_t - e_t_uncond)
            
                        
            noise_latent, pred_original_sample = self.next_noise2clean_step(e_t, timestep, noise_latent, eta, device, pred_guid, sgs_scale)
            pred_x0.append(pred_original_sample)
            
        return noise_latent, pred_x0
    
    def next_noise2clean_step(self, pred_epsilon, timestep, sample, eta, device, pred_guid=None, sgs_scale=0.0):
        alphas_cumprod = self.model.alphas_cumprod.cpu()
        prev_timestep = timestep - 20
        
        timestep = timestep
        alpha_prod_t = alphas_cumprod[timestep] if timestep >= 0 else alphas_cumprod[0]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else alphas_cumprod[0]
        sigma_t = eta * np.sqrt((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev))
        noise = sigma_t * noise_like(sample.shape, device, False)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * pred_epsilon) / alpha_prod_t ** (0.5)
        
        # Structure Guided Sampling (SGS)
        if pred_guid is not None:
            with torch.enable_grad():
                for f in range(pred_original_sample.shape[2]):
                    original_sample = pred_original_sample.requires_grad_(True)
                    merged_sample = pred_original_sample.clone()
                    merged_sample[:,:,1:] = pred_original_sample[:,:,:-1]
                    merged_sample[:,:,0] = pred_guid[:,:,-1]
                    clip_loss = F.mse_loss(pred_original_sample[:,:,:f+1], merged_sample[:,:,:f+1])
                    clip_loss.backward(retain_graph=True)
                    pred_original_sample.data -= sgs_scale * original_sample.grad

        pred_sample_direction = (1 - alpha_prod_t_prev - sigma_t**2) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction + noise
        
        return prev_sample, pred_original_sample

    def ddim_inversion(self, prev_clip, cond_embd, unc_embd, iterator, device, step_pred, lfai_scale=0.0):
        ddim_latents_all_step = self.ddim_clean2noise_loop(prev_clip, cond_embd, unc_embd, iterator, device, step_pred, lfai_scale)
        return ddim_latents_all_step

    @torch.no_grad()
    def ddim_clean2noise_loop(self, prev_clip, cond_embd, unc_embd, iterator, device, step_pred, lfai_scale=0.0):
        latent = prev_clip.clone().detach()
        
        for idx, timestep in enumerate(iterator):
            ts = torch.full((1,), timestep, device=device, dtype=torch.long)
            pred_x0 = step_pred[len(step_pred)-idx-1]
            # Dynamic Noise (DN)
            e_t = self.model.apply_model(latent, ts, unc_embd)
            range_noisy = np.linspace(start=0, stop=10, num=15)
            for i in range(1, 16):
                pivot_value = range_noisy[i-1]
                noisy_alpha = np.exp(-pivot_value)
                noise_motion = torch.empty(e_t.shape).normal_(mean=0,std=1/(1+noisy_alpha**2)).to(device)
                e_t[:,:,i] = (noisy_alpha/(math.sqrt(1+noisy_alpha**2)))*e_t[:,:,i] + noise_motion[:,:,i]

            latent = self.next_clean2noise_step(e_t, timestep, latent, idx, pred_x0, lfai_scale)
            
        return latent
    
    def next_clean2noise_step(self, pred_epsilon, timestep, sample, idx, pred_x0=None, lfai_scale=0.0):
        alphas_cumprod = self.model.alphas_cumprod.cpu()
        prev_timestep = timestep
        timestep = timestep - 20

        alpha_prod_t = alphas_cumprod[timestep] if timestep >= 0 else alphas_cumprod[0]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * pred_epsilon) / alpha_prod_t ** (0.5)

        if pred_x0 is not None:
            # Last Frame Aware Inversion (LFAI)
            with torch.enable_grad():
                original_sample = pred_original_sample.requires_grad_(True)
                frame_loss = F.mse_loss(pred_original_sample[:,:,0], pred_x0[:,:,0].detach())
                frame_loss.backward(retain_graph=True) 
                pred_original_sample.data -= lfai_scale * original_sample.grad

        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        return prev_sample
    
def view_images(images, num_rows=1, offset_ratio=0.02, save_path=None):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]
    
    if save_path is not None:
        pil_img = Image.fromarray(image_)
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        pil_img.save(f'{save_path}/{now}.png')