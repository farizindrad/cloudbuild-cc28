import os
from keras.models import load_model
import numpy as np
from skimage.transform import resize
import matplotlib.pylab as plt
from flask import Flask, request, jsonify
from tempfile import TemporaryFile, NamedTemporaryFile

ALLOWED_EXTENSIONS = {'jpg','jpeg','png'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

categories=['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___healthy','Peach___Bacterial_spot','Peach___healthy']
 
def getPrediction(img):
    # img = plt.imread('/Users/asdfg/Downloads/Mentor/Deployment/simpleh5/test3.jpg')
    resImage = resize(img,(28,28,3))
    model = load_model('./model/model.h5')
    model.make_predict_function()
    prob = model.predict(np.array([resImage],))
    sortProb = np.argsort(prob[0,:])
    return categories[sortProb[-1]]


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    # Check if file has allowed extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file extension'})

    with TemporaryFile() as temp_file:
        file.save(temp_file)
        temp_file.seek(0)

        with NamedTemporaryFile(delete=False) as temp_img_file:
            temp_img_file.write(temp_file.read())
            temp_img_file.seek(0)
            img = plt.imread(temp_img_file.name)  # Gunakan path file sementara
            
            result = getPrediction(img)
    return jsonify({'prediction': result})
    #     img = plt.imread(temp_file)
    #     result = getPrediction(img)
    

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)