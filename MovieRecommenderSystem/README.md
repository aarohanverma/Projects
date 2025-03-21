# Movie Recommender System

A web-based application built with Streamlit that recommends movies based on a user-selected title. The tool processes a movie dataset and a precomputed similarity matrix to generate personalized recommendations and fetches movie posters via The Movie Database (TMDB) API.

## High-Level Overview

The Movie Recommender System takes a movie selected by the user and computes similar movies using a precomputed similarity matrix. It then leverages movie data to fetch poster images and displays recommendations in a visually appealing layout. Key features include:
- **Recommendations:**  
  Provides a ranked list of similar movies based on the selected title.
- **Visualizations:**  
  Displays movie posters and titles for each recommendation.
- **User-Specific Selection:**  
  Allows users to choose a movie from a dropdown list to generate personalized recommendations.

This approach helps users discover movies that align with their interests by utilizing collaborative filtering techniques and integrating external API calls for poster retrieval.

## How It Works

1. **Movie Selection:**  
   Users select a movie from a dropdown list in the Streamlit sidebar. The list is populated using a preprocessed movie dataset.

2. **Similarity Calculation:**  
   The app loads a precomputed similarity matrix (from `similarity.pkl`) and a movie dataset (from `movies.pkl`). It computes the similarity scores between the selected movie and all other movies, ranking them accordingly.

3. **Poster Retrieval:**  
   For each recommended movie, the app uses the TMDB API to fetch the poster image based on the movie ID. If a poster is unavailable, a default image is shown.

4. **Display:**  
   The Streamlit app (in `app.py`) organizes and displays the recommended movies with their poster images and titles in a grid format. Recommendations are rendered dynamically after the user clicks the **Recommend** button.

## Screenshot

![screencapture-mv-rcmd-sys-av-herokuapp-2022-02-19-11_50_44](https://user-images.githubusercontent.com/97247457/154789237-66001247-fc17-4987-b03b-8856e1188c94.png)

## Local Setup

1. **Clone the Project**

   - **Open your terminal and run the following commands:**

        ```
        git clone --no-checkout https://github.com/aarohanverma/Projects.git
        cd Projects
        git sparse-checkout init --cone
        git sparse-checkout set MovieRecommenderSystem
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
        pandas
        requests
        ```
    Then install the dependencies by running:
        ```
        pip install -r requirements.txt
        ```

4. **Download Required Files:**
    - **similarity.pkl:**  
        Generate it using the accompanying notebook.
    - **movies.pkl:**  
        Ensure this pickled movie dataset is present in the project directory. (CSV files such as `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` may be used to generate it.)

5. **Run the Application:**
   Launch the Streamlit app by executing:
        ```
        streamlit run app.py
        ```

   Open your browser and navigate to [http://localhost:8501](http://localhost:8501) to view the application.

## License

This project is open-source. See the LICENSE file for more details.

---