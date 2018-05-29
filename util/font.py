import os
import numpy as np

from PIL import (Image,
                 ImageDraw,
                 ImageFont)

if os.path.exists('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'):
    MONOFONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
elif os.path.exists('/Users/josejavier/Library/Fonts/DejaVuSansMono.ttf'):
    MONOFONT_PATH = '/Users/josejavier/Library/Fonts/DejaVuSansMono.ttf'
else:
    raise ValueError("Font DejaVuSansMono.ttf not found")


def print_text(text, W, H, size=0.5):
    fg, bg = 255, 0  # white on black
    img = Image.new('L', (W, H), bg)
    draw = ImageDraw.Draw(img)
    if isinstance(size, float):
        size = int(H * size)
    font = ImageFont.truetype(MONOFONT_PATH)
    w, h = draw.textsize(text, font=font)

    draw.text(((W-w)/2, (H-h)/2), text, fg, font=font)
    return np.array(img)/255.0
