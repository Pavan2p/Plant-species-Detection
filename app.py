import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, request, render_template
import os
import uuid

# Load the trained model
model = tf.keras.models.load_model('plant_classifier_model.h5')

# Define the Flask app
app = Flask(__name__, static_url_path='/assets')

# Define the allowed file extensions for image upload
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to generate a unique filename
def generate_filename():
    unique_id = uuid.uuid4().hex
    return f'{unique_id}.jpg'

# Function to save the uploaded file
def save_file(file):
    filename = generate_filename()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath

# Home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads')
def uploads():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', error='No file selected')

    file = request.files['file']

    # Check if the file has an allowed extension
    if not allowed_file(file.filename):
        return render_template('index.html', error='Invalid file format')

    # Save the file
    filepath = save_file(file)

    # Load and preprocess the image
    img = image.load_img(filepath, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])

    # Get the class labels
    class_labels = ['Aloevera', 'Banana', 'alocasia sande', 'bilimbi', 'cantaloupe', 'cassava','coconut','corn','cucumber','curcuma','eggplant','galangal','ginger','guava','kale','longbeans','mango','melon','orange','paddy','papaya','peper chili','pineapple','pomelo','shallot','soybeans','spinach','sweet potatoes','tobacco','waterapple','watermelon']

    # Get the predicted class label
    predicted_label = class_labels[predicted_class]

    image_path = filepath
    stripped_path = image_path.replace('static/', '')
    print(stripped_path)
    # Render the result template with the uploaded image and predicted class
    return render_template('results.html', image=stripped_path, predicted_class=predicted_label)

# Set the UPLOAD_FOLDER outside the __name__ == '__main__' block
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Specify the folder where uploaded files will be saved
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Create the folder if it doesn't exist

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
