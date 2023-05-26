# Serving a Model with TensorFlow Serving
This project demonstrates the process of serving a pretrained model using TensorFlow Serving. TensorFlow Serving is a production-ready serving system that integrates with the TensorFlow stack, making it easy to deploy models for inference in a production environment.

## Key Points
1) Installation: Learn how to install TensorFlow Serving on your machine to set up the serving infrastructure.

2) Pretrained Model: Load a pretrained model that has been trained to classify dogs, cats, and birds. The model has been trained on the datasets of cats and dogs, as well as the Caltech birds dataset.

3) Model Conventions: Save the pretrained model following the conventions required by TensorFlow Serving. This ensures compatibility and smooth integration with the serving system.

4) Web Server Setup: Spin up a web server using TensorFlow Serving that can accept HTTP requests. This server will act as the interface to interact with the deployed model.

5) REST API: Interact with the model through a REST API. Send image data as requests to the server and receive predictions for the input images.

6) Model Versioning: Gain an understanding of model versioning and how TensorFlow Serving handles different versions of the model. This allows for easy updates and management of models in a production environment.

## Data Preparation
To test the deployed model, a set of test images has been provided. These images include samples from the dogs, cats, and birds classes.

Downloading Data: Download the test images using the provided link. The images are stored in a zip file.

Unzipping Data: Unzip the images and save them to a designated base directory. The images are divided into separate directories for each class.

Sample Images: Display sample images from each class to get a visual understanding of the test data.


Feel free to explore and experiment with different models and datasets to gain further insights into deploying machine learning models with TensorFlow Serving.

# Loading a Pretrained Model
In this project, we will be using a pretrained model that has been trained during Course 1 of the specialization. This model is capable of classifying images of birds, cats, and dogs with high accuracy, thanks to the use of image augmentation during training.

## Key Points
Downloading the Model Files: Download the necessary files for the pretrained model, including the protobuf file (saved_model.pb) and the variables for the model (variables.data-00000-of-00001 and variables.index). These files contain the trained weights and other essential information for the model.

1) Loading the Model: Use TensorFlow's tf.keras.models.load_model() function to load the pretrained model into memory. The model will be loaded from the saved model directory and can be assigned to a variable for further use.

2) Model Architecture: Get a summary of the loaded model to understand its architecture and the layers involved. The summary provides information about the input and output shapes of each layer, as well as the total number of trainable parameters in the model.

3) Saving the Model in SavedModel Format: To serve the model using TensorFlow Serving, we need to save the pretrained model in the SavedModel format. This format includes a protobuf file (saved_model.pb) and additional directories (assets and variables) that store the necessary information for serving the model.

4) Examining the Saved Model: Use the saved_model_cli command line utility to examine the saved model. This utility provides information about the model's signature definition, including the input and output tensors, as well as the method that can be called for inference.

The pretrained model used in this project is a simple CNN architecture. It expects input images of size (150, 150, 3), indicating colored images with a resolution of 150 by 150 pixels. The model's output is a softmax activation with 3 classes, corresponding to the classification of birds, cats, and dogs.

# Preparing Data for Inference
Before making predictions with the pretrained model, it's important to preprocess the test images to match the expected input format. The Keras ImageDataGenerator provides convenient methods to normalize pixel values, standardize image resolutions, and set the batch size for inference.

## Key Points
Normalizing Pixel Values: Use the rescale argument in the ImageDataGenerator to normalize the pixel values of the images. This ensures that the pixel values fall within the range of [0, 1], which is often required by deep learning models for optimal performance.

1) Setting Image Resolutions: Specify the target image size using the target_size argument in the ImageDataGenerator. This ensures that all images are resized to a consistent resolution that matches the input size expected by the pretrained model.

2) Batch Size for Inference: Set the batch size for inference using the batch_size argument in the ImageDataGenerator. This determines the number of images that will be processed in each batch during inference.

3) Loading Test Images: Use the flow_from_directory method of the ImageDataGenerator to load the test images from a directory. Provide the path to the directory containing the images, specify the target size and batch size, and set the class_mode to 'binary' if you have two classes or 'categorical' if you have multiple classes.

4) Label Mapping: Check the labels assigned to each class in the test generator using the class_indices attribute. This mapping provides the numerical label assigned to each class, which can be useful for interpreting the model's predictions.

5) Data and Label Shapes: Retrieve a batch of images and labels using the next function on the test generator. Verify the shapes of the data_imgs (image data) and labels (corresponding labels) arrays to ensure they match the expected shapes.

6) Sanity Check: Perform a sanity check by plotting a few images from the batch along with their true labels. This helps verify that the data preprocessing and loading process is functioning correctly.

# Serving the Model with TensorFlow Serving
In order to serve your trained model and make predictions using TensorFlow Serving, follow the steps outlined below.

## Installation
To install TensorFlow Serving, you will need to use an older version (2.8.0) as more recent versions may not be compatible with the environment. Use the following commands to install it:
!wget 'http://storage.googleapis.com/tensorflow-serving-apt/pool/tensorflow-model-server-universal-2.8.0/t/tensorflow-model-server-universal/tensorflow-model-server-universal_2.8.0_all.deb'
!dpkg -i tensorflow-model-server-universal_2.8.0_all.deb

## Starting the TensorFlow Serving Server
To start the TensorFlow Serving server and load your trained model, set the necessary environment variables and execute the following command:
python
os.environ["MODEL_DIR"] = MODEL_DIR

%%bash --bg
nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=animal_classifier \
  --model_base_path="${MODEL_DIR}" >server.log 2>&1

## Making Predictions with TensorFlow Serving
Once the server is up and running, you can make predictions by sending HTTP/REST requests to the server's REST endpoint. Follow the steps below:

1) Convert your input data (numpy arrays) to nested lists to comply with JSON format.
2) Create a JSON object with the input data as the value for the "instances" key.
3) Send a POST request to the server's endpoint using the requests library.
4) Extract the predictions from the response JSON.
5) Perform any necessary post-processing on the predictions.
Here is an example of making predictions using the REST API:

python 
import requests
import json

Convert numpy array to list (Comments)
data_imgs_list = data_imgs.tolist()

Create JSON to use in the request (Comments)
data = json.dumps({"instances": data_imgs_list})

Define headers with content-type set to JSON (Comments)
headers = {"content-type": "application/json"}

Send a POST request to the server's REST endpoint (Comments)
json_response = requests.post('http://localhost:8501/v1/models/animal_classifier:predict', data=data, headers=headers)

Parse the predictions out of the response (Comments)
predictions = json.loads(json_response.text)['predictions']

Perform any necessary post-processing on the predictions (Comments)

Example: Compute argmax (Comments)
preds = np.argmax(predictions, axis=1)

## Evaluation and Visualizing Predictions
To evaluate the performance of your model and visualize the predictions, you can compare the predicted labels with the true labels. Use the provided plot_array function to display the images along with the true and predicted labels. Here is an example:

python 
for i in range(10):
  plot_array(data_imgs[i], labels[i], preds[i])

## Note
Make sure to adjust the code snippets and parameters according to your specific setup and requirements.

## Additional Information
For more details on TensorFlow Serving and how to use it effectively, refer to the official TensorFlow Serving documentation and the TensorFlow community for any further assistance or support.


