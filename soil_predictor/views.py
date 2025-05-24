import os
import tensorflow as tf
import numpy as np
import base64
import uuid
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientnet
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from django.contrib.admin.views.decorators import staff_member_required
from .utils import convert_h5_to_savedmodel, get_model_details

# Define path to model directory
MODEL_DIR = os.path.join(settings.BASE_DIR, 'models')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def index(request):
    context = {}
    
    if request.method == 'POST' and request.FILES.get('image'):
        # Get selected model
        model_name = request.POST.get('model', 'efficientnet_model')
        
        # Process uploaded image
        image_file = request.FILES['image']
        
        if not allowed_file(image_file.name):
            context['error'] = 'Invalid file format. Please upload a PNG, JPG, or JPEG image.'
            return render(request, 'soil_predictor/index.html', context)
        
        # Save the uploaded image
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        image_path = os.path.join(upload_dir, image_file.name)
        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        
        try:
            # Get prediction
            prediction_result = predict_image(image_path, model_name)
            
            if 'error' in prediction_result:
                error_msg = prediction_result['error']
                details = error_details(error_msg)
                context['error'] = error_msg
                context['error_details'] = details
            else:
                # Create context for template
                context['prediction'] = {
                    'class': prediction_result['class'],
                    'confidence': prediction_result['confidence'] * 100  # Convert to percentage
                }
                context['model_used'] = model_name
                context['image_url'] = os.path.join(settings.MEDIA_URL, 'uploads', image_file.name)
            
        except Exception as e:
            error_msg = f'Error during prediction: {str(e)}'
            details = error_details(error_msg)
            context['error'] = error_msg
            context['error_details'] = details
    
    return render(request, 'soil_predictor/index.html', context)

@csrf_exempt
def predict_live(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        # Get selected model
        model_name = request.POST.get('model', 'efficientnet_model')
        
        # Get uploaded image
        image_file = request.FILES.get('image')
        
        if not image_file:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        # Save the captured image
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate a unique filename
        unique_filename = f"live_{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(upload_dir, unique_filename)
        
        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        
        # Get prediction
        prediction_result = predict_image(image_path, model_name)
        
        if 'error' in prediction_result:
            error_msg = prediction_result['error']
            details = error_details(error_msg)
            return JsonResponse({
                'error': error_msg,
                'error_type': details['type'],
                'suggestion': details['suggestion']
            }, status=400)
        
        # Return prediction result
        return JsonResponse({
            'class': prediction_result['class'],
            'confidence': prediction_result['confidence'] * 100,  # Convert to percentage
            'image_url': os.path.join(settings.MEDIA_URL, 'uploads', unique_filename)
        })
        
    except Exception as e:
        error_msg = f'Error during prediction: {str(e)}'
        details = error_details(error_msg)
        return JsonResponse({
            'error': error_msg,
            'error_type': details['type'],
            'suggestion': details['suggestion']
        }, status=500)

def predict_image(image_path, model_name):
    """Function to predict pH class from an image"""
    result = {}
    
    # Load the appropriate model
    model_path = os.path.join(MODEL_DIR, f'{model_name}.h5')
    if not os.path.exists(model_path):
        result['error'] = f'Model {model_name} not found.'
        return result
    
    try:
        # First attempt: Try to load with legacy Adam optimizer
        try:
            # Custom objects to handle optimizer compatibility issues
            custom_objects = {
                'Adam': tf.keras.optimizers.legacy.Adam
            }
            
            # Load model with custom objects
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Recompile the model with a compatible optimizer
            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        except Exception as load_error:
            # Second attempt: Try loading with SavedModel format instead of H5
            # This approach bypasses optimizer issues
            model_dir = os.path.join(MODEL_DIR, f'{model_name}_savedmodel')
            if os.path.exists(model_dir):
                model = tf.keras.models.load_model(model_dir, compile=False)
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            else:
                # Third attempt: Last resort - try to load with skip_optimizer
                try:
                    import h5py
                    with h5py.File(model_path, 'r') as f:
                        model_config = f.attrs.get('model_config')
                        if model_config is not None:
                            model_config = model_config.decode('utf-8')
                            model = tf.keras.models.model_from_json(model_config)
                            # Load weights only
                            model.load_weights(model_path)
                            model.compile(
                                optimizer='adam',
                                loss='categorical_crossentropy',
                                metrics=['accuracy']
                            )
                        else:
                            raise ValueError("Could not extract model config from H5 file")
                except Exception as h5_error:
                    raise ValueError(f"Failed to load model: {str(load_error)}. Alternate loading also failed: {str(h5_error)}")
        
        # Preprocess image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply appropriate preprocessing
        if 'efficientnet' in model_name:
            img_array = preprocess_input_efficientnet(img_array)
        else:
            img_array = preprocess_input_resnet(img_array)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        confidence = float(np.max(predictions))
        
        result['class'] = int(predicted_class[0])
        result['confidence'] = confidence
        
    except Exception as e:
        result['error'] = f'Error during prediction: {str(e)}'
    
    return result

@staff_member_required
def convert_model(request, model_name):
    """View to convert H5 model to SavedModel format (admin only)"""
    if model_name not in ['efficientnet_model', 'resnet_model', 'scnn_model']:
        return JsonResponse({'error': 'Invalid model name'}, status=400)
    
    success = convert_h5_to_savedmodel(model_name)
    
    if success:
        return JsonResponse({
            'status': 'success',
            'message': f'Model {model_name} successfully converted to SavedModel format'
        })
    else:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to convert model {model_name}'
        }, status=500)

@staff_member_required
def model_info(request):
    """View to get information about available models (admin only)"""
    model_names = ['efficientnet_model', 'resnet_model', 'scnn_model']
    results = {}
    
    for model in model_names:
        results[model] = get_model_details(model)
    
    return JsonResponse(results)

@staff_member_required
def admin_models(request):
    """Admin view for model management"""
    return render(request, 'soil_predictor/admin_models.html')

def error_details(error_string):
    """
    Parse error messages to provide more helpful information
    
    Args:
        error_string (str): The error message
        
    Returns:
        dict: Additional error details and suggestions
    """
    details = {
        'type': 'unknown',
        'suggestion': 'Try a different model or contact system administrator.'
    }
    
    # Check for optimizer errors
    if 'weight_decay is not a valid argument' in error_string:
        details['type'] = 'optimizer_compatibility'
        details['suggestion'] = 'This model was trained with a different TensorFlow version. ' + \
                               'Try a different model or ask an administrator to convert it to SavedModel format.'
    
    # Check for memory errors
    elif 'out of memory' in error_string.lower() or 'oom' in error_string.lower():
        details['type'] = 'memory_error'
        details['suggestion'] = 'The model is too large for available memory. Try a smaller model like SCNN instead of ResNet.'
    
    # Check for missing model files
    elif 'no such file or directory' in error_string.lower():
        details['type'] = 'missing_file'
        details['suggestion'] = 'The model file is missing. Please make sure all model files are in the correct directory.'
    
    return details
