import cv2
import glob
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image


TYPE = "unseen"
TYPE_TO_TEXT_FN = {
    "index": lambda i: f"# of subsequent writes: {i}",
    "unseen": lambda i: f"# of written datapoints: {i}",
    "forget": lambda i: f"# of 'forget' operations: {i}",
}


def add_text(img, text):
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)
    # Choose a font
    font = ImageFont.truetype("GidoleFont/Gidole-Regular.ttf", 12)
    # Draw the text
    text_location = (516, 2)
    draw.text(text_location, text, font=font)
    # Save the image
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


def to_video(img_array):
    size = (img_array[0].shape[0], img_array[0].shape[1])
    out = cv2.VideoWriter(f"demo_dir/{TYPE}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 20, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


img_array = []
glob_result = glob.glob(f"demo_dir/{TYPE}/*.png")
file_iter = sorted(glob_result, key=lambda p: int(os.path.basename(p).split('_')[-1][:-4]))
for i, filename in enumerate(file_iter):
    img = cv2.imread(filename)
    img = add_text(img, TYPE_TO_TEXT_FN[TYPE](i))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

to_video(img_array)
