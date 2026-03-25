import numpy as np
from PIL import Image

def gaussian_noise_blend(img: Image.Image, intensity: float,
                         mean: float = 127.5, std: float = 127.5,
                         seed: int | None = None) -> Image.Image:

    if seed is not None:
        rng = np.random.default_rng(seed)
        normal = rng.normal
    else:
        normal = np.random.normal

    arr = np.asarray(img).astype(np.float32)            # [H,W,3], 0..255
    H, W, C = arr.shape
    noise = normal(loc=mean, scale=std, size=(H, W, C)).astype(np.float32) 

    t = float(np.clip(intensity, 0.0, 1.0))
    out = (1.0 - t) * arr + t * noise                  
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

import numpy as np
from PIL import Image, ImageOps

def _spatial_transform_pil(img: Image.Image, t: float, rng: np.random.Generator):

    assert 0.0 <= t <= 1.0
    w, h = img.size
    out = img.copy()
    fill = (127, 127, 127)  

    if True:
    # if rng.random() < t:
        out = ImageOps.mirror(out)
    # out = ImageOps.flip(out)


    max_shrink = 0.8
    s = 1.0 - rng.uniform(0.0, max_shrink) * t 
    if s < 1.0: 
        new_w = max(1, int(round(w * s))) 
        new_h = max(1, int(round(h * s))) 
        shrunk = out.resize((new_w, new_h), resample=Image.BILINEAR) 
        canvas = Image.new(img.mode if img.mode in ("RGB", "RGBA", "L") else "RGB", (w, h), fill) 
        left = (w - new_w) // 2 
        top = (h - new_h) // 2 
        canvas.paste(shrunk, (left, top)) 
        out = canvas

    max_rot = 180.0 * t
    angle = rng.uniform(-max_rot, max_rot)
    out = out.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=fill)

    return out

def rethinking_trigger_augment(image: Image.Image, intensity: float,
                               seed: int | None = None) -> Image.Image:

    assert 0.0 <= intensity <= 1.0
    t = float(intensity)
    w, h = image.size
    rng = np.random.default_rng(seed)

    orig_arr = np.asarray(image).astype(np.float32)  # 0..255

    # 1) 对原图做可控的空间变换
    aug_pil = _spatial_transform_pil(image, t, rng)
    aug_arr = np.asarray(aug_pil).astype(np.float32)
    out_arr = np.clip(aug_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(out_arr)

from PIL import Image, ImageFilter

def gauss_blur_defense(image: Image.Image, intensity: float,
                       seed: int | None = None) -> Image.Image:

    assert 0.0 <= intensity <= 1.0
    t = float(intensity)

    img = image.convert("RGB")
    w, h = img.size

    max_radius = max(0.5, 0.06 * min(w, h))
    radius = t * max_radius

    if radius <= 1e-8:
        return img.copy()

    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))


    out = Image.blend(img, blurred, alpha=t)
    return out
