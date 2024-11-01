from flask import Flask, request, send_file, jsonify
import flask_cors
import gc
import os
import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image, ImageFile
from backgroundremover import utilities
from enhancer_engine import RealESRGAN
from backgroundremover.bg import remove
from objectRemoval_engine import SimpleLama
from PIL import Image
import io
import cv2
import glob
import numpy as np
from basicsr.utils import imwrite
from Global.detection_models import networks
from Global.detection_util.util import *
import warnings
from gfpgan import GFPGANer

warnings.filterwarnings("ignore", category=UserWarning)


app = Flask(__name__)
app.config['SECRET_KEY'] = "fhksdjhfkjshdflkj"
app.app_context()
flask_cors.CORS(app)
# ----------------------- Gan-Enhancer ---------------------------------------------

version_gan = 1.3
upscaler_gan = 2
bg_upsampler = 'realesrgan'
bg_tile = 400
ext = "auto"
suffix = None
weight_gan = 0.5

if bg_upsampler == 'realesrgan':
    if not torch.cuda.is_available():  # CPU
        import warnings
        warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                        'If you really want to use it, please modify the corresponding codes.')
        bg_upsampler = None
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        model_gan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model_gan,
            tile=bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=True)  # need to set False in CPU mode
else:
    bg_upsampler = None
arch = 'clean'
channel_multiplier = 2
model_name = 'GFPGANv1.3'
url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
# determine model paths
model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
if not os.path.isfile(model_path):
    model_path = os.path.join('gfpgan/weights', model_name + '.pth')
if not os.path.isfile(model_path):
    # download pre-trained models from url
    model_path = url
# ------------------------   Ennhancer ---------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)


#---------------------- object removal model --------------------------------------------
model_choices = ["u2net", "u2net_human_seg", "u2netp"]
bgr_model = "u2net"
alpha_matting = False
alpha_matting_foreground_threshold = 240 # The trimap foreground threshold.
alpha_matting_background_threshold = 10 # The trimap background threshold.
alpha_matting_erode_size = 10 # Size of element used for the erosion.
alpha_matting_base_size = 1000 # The image base size.
workernodes = 8#1 # Number of parallel workers
gpubatchsize = 260#2 # GPU batchsize
framerate = -1 # override the frame rate
framelimit = -1 # Limit the number of frames to process for quick testing.
mattekey = False # Output the Matte key file , type=lambda x: bool(strtobool(x)),
transparentvideo = False # Output transparent video format mov
transparentvideoovervideo = False # Overlay transparent video over another video
transparentvideooverimage = False # Overlay transparent video over another video
transparentgif = False # Make transparent gif from video
transparentgifwithbackground = False # Make transparent background overlay a background image
simple_lama = SimpleLama()

# # ----------------------------------- Restoration --------------------------------------------
restorer = GFPGANer(
        model_path=model_path,
        upscale=upscaler_gan ,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

config_gpu = 0
config_input_size = "full_size"
restore_model = networks.UNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        sync_bn=True,
        antialiasing=True,
    )
## load model
checkpoint_path = os.path.join(os.path.dirname(__file__), "Global/checkpoints/detection/FT_Epoch_latest.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
restore_model.load_state_dict(checkpoint["model_state"])
print("model weights loaded")
if config_gpu >= 0:
    restore_model.to(config_gpu)
else: 
    restore_model.cpu()
restore_model.eval()

def data_transforms(img, full_size, method=Image.BICUBIC):
    if full_size == "full_size":
        ow, oh = img.size
        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), method)

    elif full_size == "scale_256":
        ow, oh = img.size
        pw, ph = ow, oh
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == ph) and (w == pw):
            return img
        return img.resize((w, h), method)


def scale_tensor(img_tensor, default_scale=256):
    _, _, w, h = img_tensor.shape
    if w < h:
        ow = default_scale
        oh = h / w * default_scale
    else:
        oh = default_scale
        ow = w / h * default_scale

    oh = int(round(oh / 16) * 16)
    ow = int(round(ow / 16) * 16)

    return F.interpolate(img_tensor, [ow, oh], mode="bilinear")


def blend_mask(img, mask):

    np_img = np.array(img).astype("float")

    return Image.fromarray((np_img * (1 - mask) + mask * 255.0).astype("uint8")).convert("RGB")


@app.route('/', methods=['GET'])
def index():
    return jsonify(success='ok')


@app.route('/test', methods=['POST'])
def test_methods():
    print("[/test] : New data arrive")
    data = request.get_json()
    image_base64 = data['image']
    points = data["points"]
    print(len(points))
    print
    return jsonify({"message":"OK"})
    pass # end of test methods funtion

@app.route('/restoration_gan', methods=['POST'])
def restore_images_using_gan():
    print("[/restoration_gan] : New data arrive")
    image_file = request.files['image']
    img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    temp_filename = 'restore_gan.png'
    img.save(temp_filename)
    input_img = cv2.imread(temp_filename, cv2.IMREAD_COLOR)
    cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            paste_back=True,
            weight=weight_gan)
    cv2.imwrite("restore_result.png", restored_img)
    
    return send_file("restore_result.png", mimetype='image/png', as_attachment=True, download_name='restore.png')
    pass # end of endpoint function using restoring gan 

@app.route('/restoration', methods=['POST'])
def restore_ol_images():
    print("[/restoration] : New data arrive")
    image_file = request.files['image']
    img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    #detection.main(img)
    scratch_image = img
    w, h = scratch_image.size
    transformed_image_PIL = data_transforms(scratch_image, config_input_size)
    scratch_image = transformed_image_PIL.convert("L")
    scratch_image = tv.transforms.ToTensor()(scratch_image)
    scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)
    scratch_image = torch.unsqueeze(scratch_image, 0)
    _, _, ow, oh = scratch_image.shape
    scratch_image_scale = scale_tensor(scratch_image)

    if config_gpu >= 0:
        scratch_image_scale = scratch_image_scale.to(config_gpu)
    else:
        scratch_image_scale = scratch_image_scale.cpu()
    with torch.no_grad():
        P = torch.sigmoid(restore_model(scratch_image_scale))

    P = P.data.cpu()
    P = F.interpolate(P, [ow, oh], mode="nearest")

    tv.utils.save_image(
        (P >= 0.4).float(),
        os.path.join(
            "",
            "restore_mask.png",
        ),
        nrow=1,
        padding=0,
        normalize=True,
    )
    transformed_image_PIL.save(os.path.join("", "restore_orignal.png"))
    gc.collect()
    torch.cuda.empty_cache()

    msk = Image.open("restore_mask.png").convert('L')
    result = simple_lama(img, msk)
    temp_filename = 'restore_result.png'
    result.save(temp_filename)
    input_img = cv2.imread(temp_filename, cv2.IMREAD_COLOR)
    # cropped_faces, restored_faces, restored_img = restorer.enhance(
    #         input_img,
    #         paste_back=True,
    #         weight=weight_gan)
    # print(restored_img)
    # cv2.imwrite(temp_filename, restored_img)
    
    return send_file(temp_filename, mimetype='image/png', as_attachment=True, download_name='restore.png')



    pass # end of restore old images function 

@app.route('/objectremoval', methods=['POST'])
def object_removal_image():
    print("[/objectremoval] : New data arrive")
    # Get the uploaded image from the request
    image_file = request.files['image']
    mask_file = request.files['mask']
    img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    msk = Image.open(io.BytesIO(mask_file.read())).convert('L')
    result = simple_lama(img, msk)
    temp_filename = 'objectremove.png'
    result.save(temp_filename)
    return send_file(temp_filename, mimetype='image/png', as_attachment=True, download_name='objectremove.png')



@app.route('/enhanceimage', methods=['POST'])
def enhance_image():

    # Get the uploaded image from the request
    uploaded_file = request.files['image']
    img = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')

    # Perform image enhancement using the model
    sr_image = model.predict(img)

    # Save the enhanced image to a temporary file
    temp_filename = 'enhanceImage.png'
    sr_image.save(temp_filename)

    # Send the enhanced image as a response
    return send_file(temp_filename, mimetype='image/png', as_attachment=True, download_name='enhanced_image.png')


@app.route('/removebg', methods=['POST'])
def rgb():
    uploaded_file = request.files['image']
    image = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
    new_image = remove(
                    image,
                    model_name=bgr_model,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_structure_size=alpha_matting_erode_size,
                    alpha_matting_base_size=alpha_matting_base_size,
                )
    temp_filename = 'rgbImage.png'
    new_image.save(temp_filename)
    
    return send_file(temp_filename, mimetype='image/png', as_attachment=True, download_name='rgb_image.png')
    pass
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
