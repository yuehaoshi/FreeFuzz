from paddle.vision import get_image_backend

backend = get_image_backend()
print(backend)