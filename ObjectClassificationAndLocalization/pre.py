from collections import namedtuple
from PIL import Image, ImageOps, ImageDraw
import numpy as np

Bounding_Box = namedtuple('BoundingBox', ['xmin', 'ymin', 'xmax', 'ymax'])


def process_image(img_obj, for_input=False):
    target_size = (224, 224)
    img = img_obj

    # Padding image to make it Square

    width, height = img.size
    h_pad = 0
    w_pad = 0
    bonus_h_pad = 0
    bonus_w_pad = 0
    pix_diff = abs(width - height)
    if width > height:
        h_pad = pix_diff // 2
        bonus_h_pad = pix_diff % 2
    elif height > width:
        w_pad = pix_diff // 2
        bonus_w_pad = pix_diff % 2
    img = ImageOps.expand(img, (w_pad, h_pad, w_pad + bonus_w_pad, h_pad + bonus_h_pad))
    img = img.resize((target_size[0], target_size[1]))
    if for_input:
        return img
    img_data = np.array(img.getdata()).reshape(target_size[0], target_size[1], 3)
    return img_data


def image_with_bndbox(image_data, pred_bndbox=None):
    img = Image.fromarray(np.uint8(image_data)).convert('RGB')
    img1 = ImageDraw.Draw(img)
    img1.rectangle([(pred_bndbox.xmin, pred_bndbox.ymin), (pred_bndbox.xmax, pred_bndbox.ymax)], outline="blue",
                   width=3)
    return img
