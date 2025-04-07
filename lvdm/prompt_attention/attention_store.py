"""
Code of attention storer AttentionStore, which is a base class for attention editor in attention_util.py

"""

import abc
import os
import copy
import torch
import time
from lvdm.prompt_attention.util import get_time_string

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        self.cur_att_layer = 0
        self.cur_step += 1
        self.step_store = self.get_empty_store()
        # self.between_steps()
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        """I guess the diffusion of google has some unconditional attention layer
        No unconditional attention layer in Stable diffusion
store_key
        Returns:
            _type_: _description_
        """
        # return self.num_att_layers if config_dict['LOW_RESOURCE'] else 0
        return 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str, store_key):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, store_key):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            attn = self.forward(attn, is_cross, place_in_unet, store_key)
        self.cur_att_layer += 1
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, 
                 ):
        self.LOW_RESOURCE = True # assume the edit have cfg
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):
    def step_callback(self, x_t):
        x_t = super().step_callback(x_t)
        self.latents_store.append(x_t.cpu().detach())
        return x_t
    
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": [],
                "store_key": []}

    @staticmethod
    def get_empty_cross_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                }

    def forward(self, attn, is_cross: bool, place_in_unet: str, store_key):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[-2] <= 32 ** 2:  # avoid memory overhead
            # print(f"Store attention map {key} of shape {attn.shape}")
            if is_cross or self.save_self_attention:
                if attn.shape[-2] == 32**2:
                    append_tensor = attn.cpu().detach() # [8,8,1024,1024]
                else:
                    append_tensor = attn
                self.step_store[key].append(copy.deepcopy(append_tensor))
                # FIXME: Are these deepcopy all necessary?
                # self.step_store[key].append(append_tensor)
            if is_cross == False:
                if store_key is not None:
                    self.step_store["store_key"].append(copy.deepcopy(store_key))
        return attn

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store_all_step = []
        self.attention_store = {}

    def __init__(self, save_self_attention:bool=True, disk_store=False):
        super(AttentionStore, self).__init__()
        self.disk_store = disk_store
        if self.disk_store:
            time_string = get_time_string()
            path = f'./trash/attention_cache_{time_string}'
            os.makedirs(path, exist_ok=True)
            self.store_dir = path
        else:
            self.store_dir =None
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.save_self_attention = save_self_attention
        self.latents_store = []
        self.attention_store_all_step = []
