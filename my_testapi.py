from flask import Flask, request, send_file, jsonify
import flask_cors
import base64
from io import BytesIO
from objectRemoval_engine import SimpleLama
from PIL import Image, ImageFile
import cv2
import io
import numpy as np 

simple_lama = SimpleLama()

def base64toopencv(base64_string):
    im_bytes = base64.b64decode(base64_string)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img

app = Flask(__name__)
app.config['SECRET_KEY'] = "fhksdjhfkjshdflkj"
app.app_context()
flask_cors.CORS(app)



@app.route('/', methods=['GET'])
def index():
    return jsonify(success='ok')


@app.route('/removeobj', methods=['POST'])
def object_removal():
    print("/REMOVE_OBJ new request coming")
    data = request.get_json()
    base64Image= data["image"]
    base64mask= data["mask"]
    size = data["size"]
   

    print(f"Size found : {size}")

    cv_img = base64toopencv(base64Image)
    cv_mask = base64toopencv(base64mask)
    cv2.imwrite("test.png",cv_img)
    h, w, c = cv_img.shape
    cv_mask = cv2.resize(cv_mask, (w, h))
    cv2.imwrite("test_mask.png",cv_mask)


    cv_mask = cv2.cvtColor(cv_mask, cv2.COLOR_BGR2RGB) 
    cv_mask = Image.fromarray(cv_mask).convert('L') 

    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) 
    cv_img = Image.fromarray(cv_img) 

    #cv2.imwrite("test.png",cv_img)
    #cv2.imwrite("test_mask.png",cv_mask)
    

    #cv_img = Image.open("test.png")
    #cv_mask = Image.open("test_mask.png").convert('L')
    result = simple_lama(cv_img, cv_mask)

    #color_coverted = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) 
    #pil_image = Image.fromarray(color_coverted) 

    new_image = result#pil_image#Image.open("test_mask.png")
    bio = io.BytesIO()
    new_image.save(bio, "PNG")
    bio.seek(0)
    im_b64 = base64.b64encode(bio.getvalue()).decode()

    return jsonify({"bg_image":im_b64}) # end of function end point removeobj 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)