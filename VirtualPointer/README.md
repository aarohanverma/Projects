# Gesture-Controlled Mouse Using Holistic Tracking

A real-time application built with Python that leverages Mediapipe's holistic tracking to enable gesture-based mouse control. The tool processes a live webcam feed to track facial and hand landmarks and translates specific gestures into mouse actions such as clicks, double-clicks, and cursor movement.

## High-Level Overview

The Gesture-Controlled Mouse app captures video from a webcam and uses Mediapipe to extract face and hand landmarks. It then processes these landmarks to recognize gestures that control mouse operations. Key features include:
- **Gesture Recognition:** Detects eye blinks and hand gestures to trigger left/right clicks, double-clicks, and mouse dragging.
- **Holistic Tracking:** Utilizes Mediapipe's holistic module to obtain facial and hand landmarks in real time.
- **Mouse Control:** Uses the `mouse` library to simulate mouse actions based on the recognized gestures.
- **Real-Time Processing:** Continuously processes video frames to provide immediate feedback and control.

## How It Works

1. **Video Capture:**  
   The application uses OpenCV to capture a live video feed from the webcam.

2. **Frame Processing:**  
   Each frame is resized and flipped to create a mirror effect. The `HolisticTrackingModule.py` processes the frame using Mediapipe to extract landmarks for the face and hands.

3. **Gesture Detection:**  
   The `Gestures.py` module analyzes the landmarks to detect specific gestures:
   - **Eye Blinks:** Measured by the distance between key eye landmarks to trigger left and right clicks.
   - **Hand Gestures:** Evaluates the relative positions of hand landmarks (e.g., fingertip to thumb distance) to enable actions like double-click or drag-and-drop.
   
4. **Mouse Control:**  
   Based on the detected gestures, the application uses the `mouse` library to:
   - Execute single or double clicks.
   - Move the cursor according to the hand’s position.
   - Simulate mouse dragging when the appropriate gesture is maintained.

5. **Display:**  
   The processed video feed, complete with overlaid landmarks for visual feedback, is displayed in a window. This helps users verify that the system is tracking the correct landmarks and recognizing gestures accurately.

## Local Setup

1. **Clone the Project**

   - **Open your terminal and run the following commands:**
      ```
      git clone --no-checkout https://github.com/aarohanverma/Projects.git
      cd Projects
      git sparse-checkout init --cone
      git sparse-checkout set VirtualPointer
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
      opencv-python
      mediapipe
      mouse
      numpy
      ```
   Then install the dependencies by running:
      ```
      pip install -r requirements.txt
      ```
      
4. **Run the Application**

   Launch the application by executing:
      ```
      python main.py
      ```
   The application will access your webcam, process the video feed in real time, and perform mouse actions based on your gestures. To exit the application, press 'q' in the video window.

## License

This project is open-source. See the LICENSE file for more details.

---