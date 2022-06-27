import cv2
import glob
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image
import torch
import torchvision


# ffmpeg -t 60 -i demo_dir/first.avi -vf "fps=10,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
# -loop 0 demo_dir/first.gif
def add_text(img, text, text_location):
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)
    # Choose a font
    font = ImageFont.truetype("GidoleFont/Gidole-Regular.ttf", 14)
    # Draw the text
    draw.text(text_location, text, font=font, align='center')
    # Save the image
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


def plot_image_iter(trains, queries, recalls, recall_index):
    lhs_img = cv2.resize(trains[i], dsize=(128+2, 128+2))
    lhs_img = cv2.copyMakeBorder(lhs_img, 2, 2, 2, 2, 0)
    rhs_imgs = torch.stack([torch.tensor(recalls[i]) if i <= recall_index
                            else torch.tensor(queries[i]) for i in range(len(trains))],
                           dim=0).permute(0, 3, 1, 2)
    rhs_img = torchvision.utils.make_grid(rhs_imgs, nrow=10).permute(1, 2, 0).numpy()
    img = cv2.copyMakeBorder(rhs_img, 0, 0, 32+128+2+2+2, 0, 0, value=(127, 127, 127))
    lhs_y_offset = 2+64
    img[lhs_y_offset:lhs_y_offset+lhs_img.shape[0], 16:16+lhs_img.shape[1], :] = lhs_img
    img = add_text(img, "The image most recently\nstored in memory ↓", (10, lhs_y_offset-40))
    img = add_text(img,
                   "Query images and\ntheir recall results\nafter sequential\nmemory updates →",
                   (26, 266))
    return img


def to_video(img_array):
    size = (img_array[1].shape[1], img_array[0].shape[0])
    out = cv2.VideoWriter(f"demo_dir/first.avi", cv2.VideoWriter_fourcc(*'DIVX'), 4, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def list_contents(dir):
    glob_result = glob.glob(dir)
    return sorted(glob_result, key=lambda p: int(os.path.basename(p)
                                                 if '.png' not in os.path.basename(p)
                                                 else os.path.basename(p)[:-4]))


# glob_result = glob.glob("demo_dir/first/*.png")
# file_iter = sorted(glob_result, key=lambda p: int(os.path.basename(p).split('_')[-1][:-4]))
MAX_LEN = 60
trains, queries, all_recalls = [], [], []
dir_iter = list_contents("demo_dir/first/*")
for i, dirname in enumerate(dir_iter):
    if i >= MAX_LEN:
        break
    recalls = []
    file_iter = list_contents(dirname+"/*")
    for filename in file_iter:
        img = cv2.imread(filename)
        if i == MAX_LEN - 1:
            trains.append(img[2:2+64, 2:2+64, :])
            queries.append(img[2:2+64, 2+64+2:2+64+2+64, :])
        recalls.append(img[2+64+2:2+64+2+64, 2+64+2:2+64+2+64, :])
    all_recalls.append(recalls)


img_array = []
for i in range(MAX_LEN):
    img = plot_image_iter(trains, queries, all_recalls[i], i)
    img_array.append(img)

to_video(img_array)
