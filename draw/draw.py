import numpy as np
from PIL import Image, ImageDraw, ImageFont


def plot_boxes_PIL(boxes, img, color=None, labels=None, line_thickness=None):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    for box, label in zip(boxes, labels):
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), width=line_thickness, outline=tuple(color))  # plot
        if label:
            fontsize = max(round(max(img.size) / 100), 12)
            font = ImageFont.truetype("draw/Noto_Sans_KR/NotoSansKR-Regular.otf", fontsize)
            txt_width, txt_height = font.getsize(label)
            draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=tuple(color))
            draw.text((box[0] + 1, box[1] - txt_height - 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)
