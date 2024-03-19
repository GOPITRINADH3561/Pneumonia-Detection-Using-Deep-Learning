from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np

app = Flask(__name__)

dependencies = {
    'auc_roc': tf.keras.metrics.AUC()
}

verbose_name = {
    0: 'Normal Lungs',
    1: 'Pneumonic Lungs'
}

model = load_model('lung.h5', custom_objects=dependencies)

# Load a pre-trained InceptionV3 model for image categorization
pre_model = InceptionV3(include_top=True, weights='imagenet')

def check_image_category(img_path):
    # Function to use InceptionV3 to categorize the image
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    preds = pre_model.predict(img)
    # Decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    return decode_predictions(preds, top=3)[0]

def is_lung_related(categories):
    # Simple check if any of the top categories is lung-related
    for category in categories:
        if 'lung' in category[1] or 'respiratory' in category[1]:
            return True
    return False

def predict_label(img_path):
    test_image = image.load_img(img_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    predict_x = model.predict(test_image)
    classes_x = np.argmax(predict_x, axis=1)

    return verbose_name[classes_x[0]]

@app.route("/")
@app.route("/index")
@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/tests/" + img.filename
        img.save(img_path)

        # Check if the image is lung-related before prediction
        categories = check_image_category(img_path)
        if is_lung_related(categories):
            predict_result = predict_label(img_path)
        else:
            predict_result = "Not related image"

    return render_template("prediction.html", prediction=predict_result, img_path=img_path)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
