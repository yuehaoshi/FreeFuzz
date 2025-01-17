import os
import shutil
import tempfile
import numpy as np
from PIL import Image

from paddle.vision import DatasetFolder
from paddle.vision import set_image_backend

set_image_backend('pil')

def make_fake_dir():
    data_dir = tempfile.mkdtemp()

    for i in range(2):
        sub_dir = os.path.join(data_dir, 'class_' + str(i))
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        for j in range(2):
            fake_img = Image.fromarray((np.random.random((32, 32, 3)) * 255).astype('uint8'))
            fake_img.save(os.path.join(sub_dir, str(j) + '.png'))
    return data_dir

temp_dir = make_fake_dir()

pil_data_folder = DatasetFolder(temp_dir)

for items in pil_data_folder:
    break

# should get PIL.Image.Image
print(type(items[0]))

# use opencv as backend
# set_image_backend('cv2')

# cv2_data_folder = DatasetFolder(temp_dir)

# for items in cv2_data_folder:
#     break

# should get numpy.ndarray
# print(type(items[0]))

shutil.rmtree(temp_dir)