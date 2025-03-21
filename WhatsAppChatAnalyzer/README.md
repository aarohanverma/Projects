# WhatsApp Chat Analyzer

A web-based application built with Streamlit that analyzes WhatsApp chat logs. The tool processes exported chat data to generate statistics and visualizations such as message counts, word frequencies, emoji usage, and activity timelines.

## High-Level Overview

The WhatsApp Chat Analyzer takes a text file exported from WhatsApp and processes it using a custom preprocessor to extract individual messages, dates, and user information. The app then uses helper functions to compute various analytics:
- **Statistics:** Total messages, word counts, media messages, and shared links.
- **Visualizations:** Bar charts, word clouds, heatmaps, and timelines that show user activity.
- **User-Specific Analysis:** Option to view overall data or filter analysis by a specific user.

This analysis helps in understanding the chat dynamics, most active periods, and popular topics or emojis used in the conversation.

## How It Works

1. **File Upload:**  
   Users upload a WhatsApp chat file (text format) via the Streamlit sidebar.

2. **Preprocessing:**  
   The `preprocessor.py` module uses regular expressions to split the chat data into individual messages, extracting the date/time and user names. It also converts date strings into proper datetime objects and adds additional time-based columns (day, month, hour, etc.).

3. **Analysis:**  
   The `helper.py` module computes various statistics and generates visualizations. Key functionalities include:
   - Counting messages and words.
   - Creating a word cloud after filtering out common stop words (loaded from `stop_hinglish.txt`).
   - Generating plots for daily, weekly, and monthly activity.
   - Analyzing emoji usage and the frequency of common words.

4. **Display:**  
   The Streamlit app (in `app.py`) lays out the dashboard with a sidebar for file upload and user selection. Visualizations and metrics are rendered dynamically once the user clicks the **Analyze** button.

## Screenshot:

![screencapture-wca-av-herokuapp-2022-02-19-11_38_44](https://user-images.githubusercontent.com/97247457/154789017-fea5b7a8-7a23-49b3-872b-7eef5857dd23.png)

## Local Setup

1. **Clone the Project**

   - **Open your terminal and run the following commands:**
      ```
      git clone --no-checkout https://github.com/aarohanverma/Projects.git
      cd Projects
      git sparse-checkout init --cone
      git sparse-checkout set WhatsAppChatAnalyzer
      git checkout
      ```

2. **Create a Virtual Environment**

   - **For Unix or macOS:**
      ```
      python -m venv venv
      source venv/bin/activate
      ``` 
   - **For Windows:**
      ```
      python -m venv venv
      venv\Scripts\activate
      ```

3. **Install Dependencies**

   - **Ensure you have a `requirements.txt` file in your repository with the following content:**
      ```
      streamlit
      pandas
      matplotlib
      seaborn
      regex
      wordcloud
      urlextract
      emoji
      ```
   - **Then install the dependencies by running:**
      ```
      pip install -r requirements.txt
      ```

4. **Run the Application**

   - **Launch the Streamlit app by executing:**
      ```
      streamlit run app.py
      ```

   Open your browser and navigate to [http://localhost:8501](http://localhost:8501) to view the application.

## Docker Setup

1. **Build the Docker Image**

   - **Make sure you have Docker installed. Open your terminal and run:**
      ```
      docker build -t whatsapp-chat-analyzer .
      ```

2. **Run the Docker Container**

   - **Once the image is built, start the container by running:**
      ```
      docker run -p 8501:8501 whatsapp-chat-analyzer
      ```

3. **Access the Application**

   Open your browser and navigate to [http://localhost:8501](http://localhost:8501) to interact with the Chat Analyzer.

## License

This project is open-source. See the LICENSE file for more details.

---