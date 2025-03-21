# Dog vs Cat Classification and Localization App

A web-based application built with Streamlit that performs both classification and localization on images of dogs and cats. The tool processes uploaded images to determine whether the image contains a dog or a cat and to localize the detected animal by drawing a bounding box.

## High-Level Overview

The Dog vs Cat Classification and Localization App loads a pre-trained TensorFlow/Keras model (stored in `my_model.h5`) to analyze images. The app uses custom preprocessing functions to prepare the image for the model and then performs two main tasks:

- **Classification and Localization:**  
  - **Classification:** Determines if the uploaded image is of a dog or a cat.  
  - **Localization:** Predicts bounding box coordinates to mark the detected animal within the image.
  
- **Display:**  
  - The app shows the original image and the processed image with an overlaid bounding box along with a caption indicating the prediction ("Dog" or "Cat").

This approach helps users quickly identify and localize animals in images through an intuitive visual interface.

## How It Works

- **File Upload:**  
  Users upload an image (JPEG format) via the Streamlit interface.

- **Preprocessing:**  
  The `pre.py` module uses the `process_image` function to:
  - Pad and resize the image to 224×224 pixels.
  - Convert the image into a NumPy array that meets the model's input requirements.

- **Model Inference:**  
  The preprocessed image is fed into the pre-trained model (`my_model.h5`), which outputs:
  - A classification score indicating whether the image is of a dog or a cat.
  - Bounding box coordinates (xmin, ymin, xmax, ymax) for localizing the detected animal.

- **Bounding Box Drawing:**  
  The `image_with_bndbox` function from `pre.py` draws a rectangle on the original image using the predicted bounding box coordinates, visually marking the location of the animal.

- **Display:**  
  The Streamlit app (in `app.py`) displays:
  - The original uploaded image.
  - The output image with the bounding box overlay.
  - A caption that clearly shows the prediction ("Dog" or "Cat").
  All of these steps are executed when the user clicks the **Analyze** button.


## Screenshot:

![screencapture-localhost-8501-2022-02-19-11_59_46](https://user-images.githubusercontent.com/97247457/154789557-64491d28-566c-443b-a06c-e5d3416f0bf9.png)

<!-- ## Setup and Running the Application -->

## Local Setup

1. **Clone the Project**

   - **Open your terminal and run the following commands:**
      ```
      git clone --no-checkout https://github.com/aarohanverma/Projects.git
      cd Projects
      git sparse-checkout init --cone
      git sparse-checkout set ObjectClassificationAndLocalization
      git checkout
      ```

2. **Create a Virtual Environment**

   - **For Unix or macOS:**
      ```
      python3 -m venv venv
      source venv/bin/activate
      ``` 
   - **For Windows:**
      ```
      python -m venv venv
      venv\Scripts\activate
      ```

3. **Install Dependencies**

   Ensure you have a `requirements.txt` file in your repository with the following content:
      ```
      streamlit
      tensorflow
      numpy
      Pillow
      keras
      ```
   Then install the dependencies by running:
      ```
      pip install -r requirements.txt
      ```

4. **Generate the Model**

   To generate `my_model.h5` refer to the accompanying notebook or the instructions in the [IBM_ML_DL_Assignment.pdf](IBM_ML_DL_Assignment.pdf).

5. **Run the Application**

   Launch the Streamlit app by executing:
      ```
      streamlit run app.py
      ```
   Open your browser and navigate to [http://localhost:8501](http://localhost:8501) to view the application.

## Docker Setup

1. **Generate the Model**

   To generate `my_model.h5` refer to the accompanying notebook or the instructions in the [IBM_ML_DL_Assignment.pdf](IBM_ML_DL_Assignment.pdf).

2. **Build the Docker Image**

   Make sure you have Docker installed. Open your terminal and run:
      ```
      docker build -t classification-localization-app .
      ```

3. **Run the Docker Container**

   Once the image is built, start the container by running:
      ```
      docker run -p 8501:8501 classification-localization-app
      ```
      
4. **Access the Application**

   Open your browser and navigate to [http://localhost:8501](http://localhost:8501) to interact with the app.

## License

This project is open-source. See the LICENSE file for more details.

---