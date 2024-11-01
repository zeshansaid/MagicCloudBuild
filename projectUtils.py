from PIL import Image
import io
import base64
import cv2
import numpy as np 
from io import BytesIO

def base64_to_pil(im_b64): # return PIL image
    im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    return Image.open(im_file)   # img is now PIL Image object

def pil_to_base64(img): # return base64 image
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    #im_file = io.BytesIO()
    #img.save(im_file, format="PNG")
    #m_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    #im_b64 = base64.b64encode(im_bytes)
    return img_str

# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

def base64toopencv(base64_string):
    im_bytes = base64.b64decode(base64_string)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img
