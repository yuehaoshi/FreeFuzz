import numpy as np
from PIL import Image
from paddle.vision import image_load, set_image_backend

fake_img = Image.fromarray((np.random.random((32, 32, 3)) * 255).astype('uint8'))

path = 'temp.png'
fake_img.save(path)

set_image_backend('pil')

pil_img = image_load(path).convert('RGB')

# should be PIL.Image.Image
print(type(pil_img))

# use opencv as backend
# set_image_backend('cv2')

# np_img = image_load(path)
# # should get numpy.ndarray
# print(type(np_img))