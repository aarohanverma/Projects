import streamlit as st
import tensorflow as tf
import numpy as np
from pre import process_image
from pre import image_with_bndbox
from collections import namedtuple
from keras.applications.xception import preprocess_input
from PIL import Image

Bounding_Box = namedtuple('BoundingBox', ['xmin', 'ymin', 'xmax', 'ymax'])

model = tf.keras.models.load_model('my_model.h5')

st.title('Classification And Localization')
st.write('')
uploaded_file = st.file_uploader("Upload an Image of a Cat or Dog...", type=["jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = process_image(image)
    image = process_image(image, for_input=True)
    col1, col2, col3, col4, col5 = st.columns([3] + [1] * 3 + [3])

    with col1:
        st.write('')
        st.write('Input:')
        st.image(image, caption='Uploaded Image')

    with col3:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        click = st.button('Analyze')

    with col5:
        st.write('')
        if click:
            prediction = model.predict(np.array([preprocess_input(image_np)]))
            image_out = image_with_bndbox(image_np, pred_bndbox=Bounding_Box(*prediction[1][0]))
            st.write('Localized Output:')
            if prediction[0][0] > 0.5:
                st.image(image_out, caption='Prediction : Dog')
            else:
                st.image(image_out, caption='Prediction : Cat')
