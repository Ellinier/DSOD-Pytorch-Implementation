import random
from PIL import Image, ImageDraw, ImageFont
from config import cfg
# import matplotlib.pyplot as plt


def visualize_det(img, boxes, labels, scores, threshold=0.5):
    font = ImageFont.truetype("Pillow/Tests/fonts/FreeMonoBold.ttf", 24)
    draw = ImageDraw.Draw(img)
    colors = {}

    colors[0] = (205,201,201)
    colors[1] = (255,255,0)
    colors[2] = (0,0,0)
    colors[3] = (255,250,205)
    colors[4] = (25,25,112)
    colors[5] = (0,191,255)
    colors[6] = (85,107,47)
    colors[7] = (240,230,140)
    colors[8] = (173,255,47)
    colors[9] = (255,0,0)
    colors[10] = (138,43,226)
    colors[11] = (255,20,147)
    colors[12] = (255,255,255)
    colors[13] = (148,0,211)
    colors[14] = (184,134,11)

    if boxes is None:
        return img

    for i,box in enumerate(boxes):
        box[::2] *= img.width
        box[1::2] *= img.height
        label = int(labels[i].numpy()[0])
        score = scores[i].numpy()[0]
        if score > threshold:
            if label not in colors:
                colors[label] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            coord = list(box)
            draw.rectangle(coord, outline=colors[label])
            draw.rectangle([coord[0]-1, coord[1]-1, coord[2]+1, coord[3]+1],outline=colors[label])
            draw.rectangle([coord[0]+1, coord[1]+1, coord[2]-1, coord[3]-1],outline=colors[label])
            str = '{}'.format(cfg.CLASSES[label])
            str = '{}\n{:.2f}'.format(cfg.CLASSES[label], score)
            draw.text((box[0], box[1]), str, colors[label],font=font)
    return img

