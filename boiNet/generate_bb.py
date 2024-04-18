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
df.to_csv(output_csv, index=False)


