from PIL import Image
import os
from .image_util import IMAGE_EXTENSION
from einops import rearrange
import torch
import numpy as np

def get_image_list(path):
    images = []
    for file in sorted(os.listdir(path)):
        if file.endswith(IMAGE_EXTENSION):
            images.append(file)
    return images

def load_frame(index, path, images):
    image_path = os.path.join(path, images[index])
    return Image.open(image_path).convert("RGB")

def tensorize_frames(frames):
    frames = rearrange(np.stack(frames), "f h w c -> c f h w")
    return torch.from_numpy(frames).div(255) * 2 - 1

def short_size_scale(images, size):
    h, w = images.shape[-2:]
    short, long = (h, w) if h < w else (w, h)

    scale = size / short
    long_target = int(scale * long)

    target_size = (size, long_target) if h < w else (long_target, size)

    return torch.nn.functional.interpolate(
        input=images, size=256, mode="bilinear", antialias=True
    )

def img2frame(path):
    # path = "./teaser_car"
    images = get_image_list(path)
    frames = [load_frame(i, path, images) for i in range(len(images))]
    tensorize_frame = tensorize_frames(frames)
    resized_frames = torch.nn.functional.interpolate(input=tensorize_frame, size=256, mode="bilinear", antialias=True)
    return resized_frames.unsqueeze(0)

if __name__ == "__main__":
    print(img2frame().shape)