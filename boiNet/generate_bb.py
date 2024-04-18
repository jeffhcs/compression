import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN
import os
from tqdm import tqdm
import numpy as np

mtcnn = MTCNN(keep_all=True)
image_dir = '../celeba/img_align_celeba'
output_csv = 'test_set_bb.csv'
results = []

start, end = 20001, 20001 + 10000

# Phase 1: Fit bounding boxes over each image
for i in tqdm(range(start, end)):
    file_name = (6 - len(str(i))) * '0' + str(i) + '.jpg'
    file_path = os.path.join(image_dir, file_name)
    image = Image.open(file_path)
    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        continue
    elif len(boxes) > 1:
        box_areas = [(box, (box[2] - box[0]) * (box[3] - box[1])) for box in boxes]
        box = max(box_areas, key=lambda x: x[1])[0]
        box = np.round(box.astype(float)).astype(int)
        results.append([file_name, *box])
    else:
        box = np.round(boxes[0].astype(float)).astype(int)
        results.append([file_name, *box])

df = pd.DataFrame(results, columns=['filename', 'x1', 'y1', 'x2', 'y2'])

# Phase 2: Resize bounding boxes to a consistent size
# Define new width and height
new_width = 84
new_height = 92

# Calculate the center points of the original bounding boxes
df['center_x'] = (df['x1'] + df['x2']) / 2
df['center_y'] = (df['y1'] + df['y2']) / 2

# Adjust the vertical center by lowering it 10 pixels
df['adjusted_center_y'] = df['center_y'] + 10

# Calculate new x1, y1, x2, y2 based on the new dimensions and adjusted center points
df['norm_x1'] = df['center_x'] - new_width / 2
df['norm_y1'] = df['adjusted_center_y'] - new_height / 2
df['norm_x2'] = df['center_x'] + new_width / 2
df['norm_y2'] = df['adjusted_center_y'] + new_height / 2

# Create a new dataframe with the filename and normalized coordinates
norm_df = df[['filename', 'norm_x1', 'norm_y1', 'norm_x2', 'norm_y2']]
norm_df.columns = ['filename', 'x1', 'y1', 'x2', 'y2']  # Rename columns to match the original format

norm_df.to_csv(output_csv, index=False)