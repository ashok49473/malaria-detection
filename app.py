from flask import Flask, render_template, request
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('vgg.h5')

def predict_class(img):
    x = image.img_to_array(img)
    x = x/255.0
    x = np.expand_dims(x,axis=0)
    proba = model.predict(x)[0][0]

    y = "Uninfected" if proba > 0.5 else "Parasitized"

    if y == "Parasitized" : proba = 1-proba

    return y, round(proba*100, 3)

app = Flask(__name__)

UPLOAD_FOLDER = "static/images/"

# Remove some images from folder if folder length > MAX_SIZE
'''
folder_lenth = len(os.listdir(UPLOAD_FOLDER))
MAX_SIZE = 5
for img in os.listdir(UPLOAD_FOLDER):
	if folder_lenth < MAX_SIZE:
		break
	else:
		os.remove(os.path.join(UPLOAD_FOLDER, img))
'''
@app.route("/", methods=['GET', 'POST'])
def upload_predict():
	if request.method == "POST":
		image_file = request.files['image']
		if image_file:
			loc = os.path.join(UPLOAD_FOLDER, image_file.filename)
			image_file.save(loc)
			img = image.load_img(loc,target_size=(150, 150))
			y, proba = predict_class(img)
			return render_template("home.html", prediction=y, img_loc = loc, proba=proba)
	return render_template("home.html")


@app.route("/about")
def about():
	return render_template("about.html")

if __name__ == "__main__":
	app.run(debug=True, port=8000)