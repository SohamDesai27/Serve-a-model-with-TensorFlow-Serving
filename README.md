# Serve-a-model-with-TensorFlow-Serving
To serve a model with TensorFlow Serving, you need to follow these steps:

1) Install TensorFlow Serving: You can install TensorFlow Serving using the following commands:
!wget 'http://storage.googleapis.com/tensorflow-serving-apt/pool/tensorflow-model-server-universal-2.8.0/t/tensorflow-model-server-universal/tensorflow-model-server-universal_2.8.0_all.deb'
!dpkg -i tensorflow-model-server-universal_2.8.0_all.deb

2) Save the model in the SavedModel format: TensorFlow Serving requires the model to be saved in the SavedModel format. You can save your model using the tf.keras.models.save_model() function. Make sure to specify the correct export path:
import tempfile

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print(f'export_path = {export_path}\n')

# Save the model
tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

3) Start TensorFlow Serving: After installing TensorFlow Serving and saving the model, you can start running TensorFlow Serving using the following command:
!tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=my_model \
  --model_base_path="/path/to/saved_model"
Make sure to replace /path/to/saved_model with the actual path where you saved the model.

4) Interact with the model via a REST API: Once TensorFlow Serving is running, you can interact with the model by sending HTTP requests to the REST API endpoint. For example, you can use the requests library in Python to send POST requests to the endpoint:
import requests

# Example inference request
input_data = {
    "instances": [
        {"conv2d_input": data_imgs[0].tolist()},
        {"conv2d_input": data_imgs[1].tolist()},
        {"conv2d_input": data_imgs[2].tolist()},
        # Add more instances as needed
    ]
}

response = requests.post('http://localhost:8501/v1/models/my_model:predict', json=input_data)
predictions = response.json()

# Process the predictions
# ...
Make sure to replace localhost:8501 with the appropriate hostname and port if TensorFlow Serving is running on a different machine or port.

Note: Before starting TensorFlow Serving, ensure that you have installed it and have the necessary permissions to run it. Additionally, adjust the code snippets provided to match your specific model and setup.
