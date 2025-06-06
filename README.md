# Soil pH Predictor

A Django web application for predicting soil pH levels using machine learning models.

## Setup Instructions

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Make sure you have the trained models in the `models/` directory:
   - efficientnet_model.h5
   - resnet_model.h5
   - scnn_model.h5

3. Run the Django development server:
```
python manage.py runserver
```

4. Access the application at http://127.0.0.1:8000/

## Features

- Upload soil images to predict pH levels
- Choose between different ML models:
  - EfficientNet
  - ResNet
  - SCNN
- View prediction results with confidence scores #   d j a n g o - p h - t a n a h  
 