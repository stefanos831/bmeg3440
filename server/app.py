from flask import Flask, request, jsonify
import requests
import keras
import json
from flask_cors import CORS
import librosa
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image

app = Flask(__name__)
CORS(app)

audio_path = "C:/Users/cherr/Downloads/BMEG3440/Sounds/original/TD16/1.WAV" 
@app.route('/predict', methods=['GET','POST'])
# def trial ():
#     if request.method == 'POST':
#         # data = request.json
#         # file = request.files('file')
#         # print(file)
#         data = request.get_data('data')
#         data = data.decode()
#         data = json.loads(data)
#         print(data)
#         print('data',data_data['userid'])
#         return ('data received')
#     else:
#         return ('404')
    
def predict_route():
    if request.method == 'POST':
        data = request.get_data('data')
        print(data)
        data = data.decode()
        data = json.loads(data)
        return data['userid']
        # print(type(data))
    # model = keras.models.load_model('C:/Users/cherr/Downloads/3440/model.keras')
    # if data:
    #     print(data['userid'])
    #     feature = extract_feature(np.array(data['userid']))
    #     prediction = model.predict(feature)
    #     if prediction >= 0.5:
    #         prediction = 1
    #     elif prediction <0.5:
    #         prediction = 0        
    #     return jsonify({'result':prediction})
    else: return 'No Input Received'

def extract_feature(audio):
    model = VGG16(weights='imagenet', include_top=False)
    # audio, sr = librosa.load(audio_path)
    # Convert audio to spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=16000)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Resize the spectrogram to match the input size expected by VGG16
    desired_shape = (224, 224)
    resized_spectrogram = np.expand_dims(spectrogram, axis=-1)
    resized_spectrogram = np.repeat(resized_spectrogram, 3, axis=-1)  # Convert to 3-channel image

    # Preprocess the spectrogram
    x = image.array_to_img(resized_spectrogram)
    x = x.resize((desired_shape[1], desired_shape[0]))  # Resize image to desired shape
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Extract features using the pre-trained VGG16 model
    features = model.predict(x)
    return features


@app.route('/reverse_geocode', methods=['GET'])
def reverse_geocode():
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    # Google Maps Geocoding API endpoint
    api_url = f'https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key=AIzaSyBDAbly995M-U5TWDROACbpmfFOY7t_iho'
    #api_url = f'https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lng}'
    response = requests.get(api_url)

    data = response.json()

    if data['status'] == 'OK':
        result = data['results'][0]
        formatted_address = result['formatted_address']
        location_type = 'street'

        for component in result['address_components']:
            types = component['types']
            if 'premise' in types or 'building' in types:
                location_type = 'building'
                break
            elif 'park' in types or 'natural_feature' in types:
                location_type = 'park'
                break
            elif 'body_of_water' in types or 'natural_feature' in types:
                location_type = 'sea'
                break
            elif 'mountain' in types or 'natural_feature' in types:
                location_type = 'mountain'
                break
            elif 'establishment' in types and 'shopping_mall' in types:
                location_type = 'shopping_mall'
                break


        return f"{formatted_address} | {location_type}"

    else:
        return {'error': 'Unable to retrieve address'}
if __name__ == '__main__':
    app.run(host="0.0.0.0")

