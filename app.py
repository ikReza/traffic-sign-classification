import streamlit as st
import numpy as np
from tensorflow.keras import models
import cv2

sign_labels = {0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
     3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (30km/h)",
     6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
     9: "No passing", 10: "No passing for vehicles over 3.5 metric tons",
     11: "Right-of-way at the next intersection", 12: "Priority road", 13: "Yield", 14: "Stop",
     15: "No vehicles", 16: "Vehicles over 3.5 metric tons prohibited", 17: "No entry",
     18: "General caution", 19: "Dangerous curve to the left", 20: "Dangerous curve to the right",
     21: "Double curve", 22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on the right",
     25: "Road work", 26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
     29: "Bicycle crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
     32: "End of all speed and passing limits", 33: "Turn right ahead", 34: "Turn left ahead",
     35: "Ahead only", 36: "Go straight or right", 37: "Go straight or left", 38: "Keep right",
     39: "Keep left", 40: "Roundabout mandatory", 41: "End of no passing",
     42: "End of no passing by vehicles over 3.5 metric tons"}

st.set_option("deprecation.showfileUploaderEncoding", False)
@st.cache(allow_output_mutation=True)
def load_model():
    model = models.load_model("./contents/traffic_sign.hdf5")
    
    return model

model = load_model()

st.title("Traffic sign classification ⛔🚳")

def show_classes():
    li = []
    for key in sign_labels:
        li.append(str(key+1) + ": " + sign_labels[key])
    
    return li

st.sidebar.selectbox("43 classes of images", show_classes())

file = st.file_uploader("Please upload an image", type=["jpg", "png", "jpeg"])

def import_and_predict(img_data, model):
    size = (32, 32)
    resized_img = cv2.resize(img_data, size)
    gray_img = np.sum(resized_img/3, axis=2, keepdims=True)
    norm_img = (gray_img - 128)/128
    reshaped_norm_img = np.array([norm_img])
    prediction = model.predict_classes(reshaped_norm_img)
    
    return sign_labels[int(prediction)]
    

if file is None:
    st.text("Please upload an image")
else:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    #img = cv2.imread(img)
    st.image(img, use_column_width=True, channels="BGR")
    prediction = import_and_predict(img, model)
    st.success(prediction)