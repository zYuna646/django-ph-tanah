import os
import tensorflow as tf
from django.conf import settings

def convert_h5_to_savedmodel(model_name):
    """
    Convert an H5 model to SavedModel format to avoid optimizer compatibility issues
    
    Args:
        model_name (str): Name of the model file without extension (e.g., 'efficientnet_model')
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Paths
        model_dir = os.path.join(settings.BASE_DIR, 'models')
        h5_path = os.path.join(model_dir, f'{model_name}.h5')
        savedmodel_dir = os.path.join(model_dir, f'{model_name}_savedmodel')
        
        # Check if H5 model exists
        if not os.path.exists(h5_path):
            print(f"Model file {h5_path} not found.")
            return False
        
        # Check if SavedModel already exists
        if os.path.exists(savedmodel_dir):
            print(f"SavedModel directory {savedmodel_dir} already exists.")
            return True
        
        # Load model without optimizer to avoid issues
        model = tf.keras.models.load_model(h5_path, compile=False)
        
        # Recompile with a generic optimizer
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save in SavedModel format
        tf.saved_model.save(model, savedmodel_dir)
        print(f"Model successfully converted and saved to {savedmodel_dir}")
        return True
        
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        return False

def get_model_details(model_name):
    """
    Get details about a model
    
    Args:
        model_name (str): Name of the model file without extension
    
    Returns:
        dict: Dictionary containing model details
    """
    details = {
        'name': model_name,
        'exists': False,
        'type': None,
        'size': None,
        'layers': None
    }
    
    try:
        # Check H5 model
        h5_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}.h5')
        if os.path.exists(h5_path):
            details['exists'] = True
            details['type'] = 'H5'
            details['size'] = os.path.getsize(h5_path) / (1024 * 1024)  # Size in MB
            
            # Try to load model to get layer count
            model = tf.keras.models.load_model(h5_path, compile=False)
            details['layers'] = len(model.layers)
        
        # Check SavedModel
        savedmodel_dir = os.path.join(settings.BASE_DIR, 'models', f'{model_name}_savedmodel')
        if os.path.exists(savedmodel_dir):
            details['exists'] = True
            details['type'] = 'SavedModel'
            
            # Calculate directory size
            total_size = 0
            for path, dirs, files in os.walk(savedmodel_dir):
                for f in files:
                    fp = os.path.join(path, f)
                    total_size += os.path.getsize(fp)
            
            details['size'] = total_size / (1024 * 1024)  # Size in MB
            
            # Try to load model to get layer count
            if 'layers' not in details or details['layers'] is None:
                model = tf.keras.models.load_model(savedmodel_dir, compile=False)
                details['layers'] = len(model.layers)
                
    except Exception as e:
        details['error'] = str(e)
    
    return details 