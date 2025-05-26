# docker pull tensorflow/serving
# docker run -t --rm -p 8501:8501 -v C:\Users\annla\Documents\Learning-Materials\Projects\Tomato_Disease_Classification:/Tomato_Disease_Classification tensorflow/serving --rest_api_port=8501 --model_config_file=/Tomato_Disease_Classification/models.config


import tensorflow as tf
import os
import tensorflow as tf
import os

# 1. Get the directory of the current script (draft.py)
# Assuming draft.py is in the project root: /YourProject/draft.py
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construct the absolute path to your original .h5 model
# This should point to: C:\Users\annla\Documents\Learning-Materials\Projects\Tomato_Disease_Classification\saved_models\1\tomatoes_model.h5
h5_model_path = os.path.join(script_dir, 'saved_models', '1', 'tomatoes_model.h5')

# 3. Construct the absolute path for the NEW SavedModel format output directory
# This should be: C:\Users\annla\Documents\Learning-Materials\Projects\Tomato_Disease_Classification\saved_models\tomatoes_model\1
# We'll create the 'tomatoes_model' directory and the '1' version directory inside it.
saved_model_output_base_path = os.path.join(script_dir, 'saved_models', 'tomatoes_model')
saved_model_output_version_path = os.path.join(saved_model_output_base_path, '1') # The '1' here is the version number

# --- Error Handling (optional but good practice) ---
if not os.path.exists(h5_model_path):
    print(f"Error: .h5 model not found at: {h5_model_path}")
    exit() # Exit if the source file doesn't exist
# --- End Error Handling ---

print(f"Attempting to load .h5 model from: {h5_model_path}")
try:
    model = tf.keras.models.load_model(h5_model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load .h5 model. Error: {e}")
    exit()

# Ensure the output directory exists before saving
# Use exist_ok=True to avoid error if directory already exists
os.makedirs(saved_model_output_version_path, exist_ok=True)

print(f"Attempting to save model to SavedModel format at: {saved_model_output_version_path}")
try:
    # --- THIS IS THE KEY CHANGE ---
    model.export(saved_model_output_version_path) # Removed save_format='tf'
    # ----------------------------
    print(f"Model converted and saved successfully to: {saved_model_output_version_path}")
except Exception as e:
    print(f"Failed to save model in SavedModel format. Error: {e}")