# Emotion Recognition - Backend

## Overview

## Steps did in this project :

### Installed things :

pip install flask

pip install scikit-learn

pip install deepface

pip install tf-keras


check for the application is there or not:
flask shell

If you didn't see anything running(set the FLASK_APP environment variable to your application factory):
export FLASK_APP=__init__:create_app



### To run the main file:

flask run

### Now to test the api:

Go to postman

POST

http://127.0.0.1:5000/getemotion-svm

Go to Body: Key as image, value put the Image file

### Things to install for render

For render to run your app:
pip install gunicorn

Need requirements file to render to install if needed:
pip freeze > requirements.txt 

Now push your code to git

While deploying add these to render:
pip install -r requirements.txt

start command:
gunicorn run:app


## Project Struture is :

### Happy Coding!

---

**Note:** This is a Backend with the Flask.
