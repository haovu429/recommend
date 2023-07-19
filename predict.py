from flask import Flask, request, jsonify
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import requests
from io import BytesIO
import cloudinary
import cloudinary.api

app = Flask(__name__)

# Load the necessary data and models
feature_list = np.array(pickle.load(open('embeddings1.pkl','rb')))
filenames = pickle.load(open('ilenames.pkl','rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def feature_extraction(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices
# # Configure Cloudinary credentials
# cloudinary.config( 
#   cloud_name = "dqhesbxwu", 
#   api_key = "755286768728456", 
#   api_secret = "ApXI3oIPiIHaMCxxwoiE3Dz-SHk" 
# )
# Configure Cloudinary credentials
cloudinary.config( 
  cloud_name = "duae6ktzt", 
  api_key = "614158457479499", 
  api_secret = "VXSpKbvdBKuYcH8CFfGx0AGOHYQ" 
)

@app.route('/recommend', methods=['POST'])
def recommend_image():
    #print(request.form['image_url'])
    if 'image_url' not in request.form:
        return jsonify({'error': 'No image URL provided'})

    image_url = request.form['image_url']
    
    # Download the image from the URL
    try:
        response = requests.get(image_url)
        # img = Image.open(BytesIO(response.content))
        img = Image.open(BytesIO(response.content)).convert('RGB')

    except:
        return jsonify({'error': 'Failed to download the image from the provided URL'})

    features = feature_extraction(img)
    indices = recommend(features)

    recommended_images = [get_cloudinary_image_url(filenames[idx]) for idx in indices[0]]

    return jsonify({'recommended_images': recommended_images})

def get_cloudinary_image_url(filename):
    # Tách tên tệp tin từ đường dẫn local
    file_name = os.path.basename(filename)
    print(file_name)
    # Lấy URL của tệp tin từ Cloudinary
    test_name = "samples/animals/lenna"
    resource = cloudinary.api.resource(get_path(file_name))
    image_url = resource["secure_url"]
    return image_url

# def find_resource_by_filename(filename):
#     resources = cloudinary.api.resources_by_tag("clothes", context=True)
#     print(resources)
#     for resource in resources["resources"]:
#         print(resource["filename"])
#         if resource["filename"] == filename:
#             return resource["secure_url"]
#     return None
def get_path(path):
    image_path = path.split(".")
    print(image_path)
    # final_path = "images/" + image_path[0]
    final_path = image_path[0]
    return final_path
if __name__ == '__main__':
    app.run(debug=True)