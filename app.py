from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import os
import re
from werkzeug.utils import secure_filename
from pickle import load
from numpy import argmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import load_img
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import cv2
import numpy as np


app = Flask(__name__)
app.secret_key = 'jshvk63787982ujkrwhvfi'

# Set the upload folder
UPLOAD_FOLDER = 'static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Regex for email validation
EMAIL_REGEX = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

# SQLite database setup
DATABASE = 'users.db'

# Function to initialize the database and create the users table if it doesn't exist
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

# Function to validate email format
def validate_email(email):
    if re.match(EMAIL_REGEX, email):
        return True
    return False

# Function to add a user to the database
def add_user_to_db(username, email, password):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', (username, email, password))
        conn.commit()

# Function to get user by username and password from the database
def get_user_from_db(username, password):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        return c.fetchone()

@app.route("/")
def homepage():
    return render_template('index.html')

@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/caption")
def caption():
    return render_template('caption.html')

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm-password']

        # Check if the username already exists
        if get_user_from_db(username, password):
            flash("Username already exists!", "danger")
            return redirect(url_for('signup'))

        # Validate email
        if not validate_email(email):
            flash("Invalid email format.", "danger")
            return redirect(url_for('signup'))

        # Check if password and confirm password match
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('signup'))

        # Store username, email, and password in the database
        add_user_to_db(username, email, password)
        flash("Signup successful! You can now login.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        print(username)

        # Check if the username exists and password matches
        user = get_user_from_db(username, password)
        if user:
            session['username'] = username  # Store the username in session
            flash("Login successful!", "success")
            return redirect(url_for('caption'))  # Redirect to the caption upload page
        else:
            flash("Invalid username or password.", "danger")

    return render_template('login.html')

@app.route("/logout")
def logout():
    session.pop('username', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if 'username' not in session:
        flash("Please login to access this page.", "warning")
        return redirect(url_for('login'))

    description = None
    p = None

    # Handle GET request (no caption generated yet)
    if request.method == "GET":
        return render_template('upload.html', cp=description, src=p)

    # Handle POST request (generate caption after file upload)
    if request.method == "POST" and 'photo' in request.files:
        file = request.files['photo']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            p = file_path

        description = generate_caption(p)

    return render_template('upload.html', cp=description, src=p)


# Helper functions for caption generation (based on your original app)
# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

def extract_feature(filename):
    # Load the VGG16 model
    model = VGG16()
    # Remove the last layer to get features
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # Load and preprocess the image
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    # Extract features
    feature = model.predict(image, verbose=0)
    return feature

def generate_caption(image_path):

    tokenizer = load(open('model/tokenizer.pkl', 'rb'))
    model = load_model('model/model.h5')

    # Load and preprocess the image using OpenCV
    image = cv2.imread(image_path)
    image_array = cv2.resize(image, (224, 224))  # Resize image to match VGG16 input shape
    image_array = image_array.astype('float32') / 255  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Extract features from the image
    photo = extract_feature(image_path)

    # Generate caption
    caption = generate_desc(model, tokenizer, photo, max_length=34)

    return caption


if __name__ == "__main__":
    # Initialize the database and create the table if it doesn't exist
    init_db()

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=True)
