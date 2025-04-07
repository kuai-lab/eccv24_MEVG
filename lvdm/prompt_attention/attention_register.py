"""
register the attention controller into the UNet of stable diffusion
Build a customized attention function `_attention'
Replace the original attention function with `forward' and `spatial_temporal_forward' in attention_controlled_forward function
Most of spatial_temporal_forward is directly copy from `video_diffusion/models/attention.py'
TODO FIXME: merge redundant code with attention.py
"""

from einops import rearrange, repeat
import torch
from torch import nn, einsum
import torch.nn.functional as F

from lvdm.models.modules.util import (
    exists,
    default,
)

def register_attention_control(model, controller):
    "Connect a model with a controller"
    def attention_controlled_forward(self, place_in_unet, attention_type='cross'):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def _attention(q, k, v, is_cross, attention_mask=None, attn_map=None, store_key=None):
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            attn = sim.softmax(dim=-1)

            #-----------------------------------control attention map
            if attn_map is not None:
                attn_map = attn_map.to(attn.device)

                # if is_cross == True:
                #     with torch.enable_grad():
                #         attn_map = attn_map.detach().requires_grad_(True)
                #         loss = F.mse_loss(attn[:8,:,:], attn_map[:,:,:])
                #         loss.backward(retain_graph=True)
                #         attn.data[:8] = attn.data[:8] - 3000 * attn_map.grad

                # if is_cross == True:
                #     attn[:8,:,1:4] = attn_map[:,:,1:4]
                #     attn[:8,:,5:6] = attn_map[:,:,5:6]
                #     attn[:8,:,8:9] = attn_map[:,:,7:8]
                # else:
                # if is_cross == True:
                #     with torch.enable_grad():
                #         attn_map = attn_map.detach().requires_grad_(True)
                #         loss = F.mse_loss(attn[:8,:,:], attn_map[:,:,:])
                #         loss.backward(retain_graph=True)
                #         attn.data[:8,:,:] = attn.data[:8,:,:] - 3000 * attn_map.grad[:,:,:]
                #     pass
            #-----------------------------------control attention map
            controller(attn[-8:,:,:], is_cross, place_in_unet, store_key)
            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
            return out
        
        def forward(x, context=None, mask=None, attn_map=None, prev_key=None):
            is_cross = context is not None
            store_key = None
            context = default(context, x)
            h = self.heads

            q = self.to_q(x)
            k = self.to_k(context)
            v = self.to_v(context)

            if is_cross == False:
                store_key = k[-1,:,:].clone()
            
            # if prev_key is not None and is_cross is False:
            # if prev_key is not None:
            #     with torch.enable_grad():
            #         prev_key = prev_key.detach().requires_grad_(True)
            #         loss = F.mse_loss(k[0,:,:], prev_key)
            #         loss.backward(retain_graph=True)
            #         k.data[0,:,:] = k.data[0,:,:] - 30 * prev_key.grad
            #     v = k.clone()
                # k[0] = prev_key
                # v[0] = prev_key

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            hidden_states = _attention(q, k, v, is_cross=is_cross, attention_mask=mask, attn_map=attn_map, store_key=store_key)

            # linear proj
            hidden_states = to_out(hidden_states)
            return hidden_states

        if attention_type == 'CrossAttention':
            return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()
    
    def register_recr(net_, count, place_in_unet):
        # print(net_)
        if net_.__class__.__name__ == 'CrossAttention':
            # for p in net_.parameters():
            #     print(p.shape)
            net_.forward = attention_controlled_forward(net_, place_in_unet, attention_type = net_.__class__.__name__ )
            return count + 1
        elif hasattr(net_, 'children'):
            for net in net_.named_children():
                if net !='attn_temporal':
                    count = register_recr(net[1], count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if net[1].__class__.__name__ == 'DiffusionWrapper': 
            if net[1].diffusion_model.input_blocks:
                # print("----------------LEN>", len(net[1].diffusion_model.input_blocks))
                for idx_in_block in range(len(net[1].diffusion_model.input_blocks)):
                    cross_att_count += register_recr(net[1].diffusion_model.input_blocks[idx_in_block], 0, "down")
                    # print(Adsfasdfasdf)
            if net[1].diffusion_model.output_blocks:
                for idx_in_block in range(len(net[1].diffusion_model.output_blocks)):
                    cross_att_count += register_recr(net[1].diffusion_model.output_blocks[idx_in_block], 0, "up")
            if net[1].diffusion_model.middle_block:
                cross_att_count += register_recr(net[1].diffusion_model.middle_block, 0, "mid")
    
    # print(f"Number of attention layer registered {cross_att_count}")
    controller.num_att_layers = cross_att_count