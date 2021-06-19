# Malaria Infection Detector : Project Overview🎯
### A Machine Learning based web application to detect the malaria infection in the given cell image.
 - Deep Learning Model - Customized version of VGG16 Model Architecture
 - The model was trained on 22048 images belonging to 2 classes (Uninfected / Parasitized) and validated on 5510 images.
 - Training accuracy - 0.972 **&** Validation accuracy - 0.947
 - Developed a user interface using Flask to upload image and see results.

## Code and Resources 👨🏻‍💻
  - Data set : [Malaria Cell Images](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)
  - Code : [Jupyter Notebook!](https://www.kaggle.com/ashokkumarpalivela/malaria-detection-app-with-tensorflow-2)
  - Useful Blog : [How to Develop VGG Model from Scratch in Keras](https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/)

## Tools used🛠
 - Python 3.7
 - Libraries
   - TensorFlow 2
   - Flask
   - Numpy
 - HTML
 - Bootstrap
 
 ## Run this application☢
 - Step 1: Download zip file and extract or clone this repository<br>
 `git clone https://github.com/ashok49473/malaria-detection.git`

- Step 2: Open your command promt<br>
change directory : `cd desktop/malaria-detection`

 - Step 3: Install dependencies<br>
`pip install -r requirements.txt`

 - Step 4: Run the application<br>
`python app.py`

 - Step 5: Copy the *url* from the terminal and open it in your favourite browser.

## Demo
 - Home Page<br>
  <img src="https://github.com/ashok49473/malaria-detection/blob/main/static/images/Home.png" width="75%"><br>
  
 - Result<br>
 <img src="https://github.com/ashok49473/malaria-detection/blob/main/static/images/result.png" width="75%">
