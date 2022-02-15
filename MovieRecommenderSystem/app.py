import streamlit as st
import pickle
import requests

movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))


def fetch_poster(movie_id):
    response = requests.get(
        'https://api.themoviedb.org/3/movie/{0}?api_key=21734bbaa71c03ba960463a099b62da7'.format(movie_id))
    data = response.json()
    if data['poster_path'] is not None:
        return 'https://image.tmdb.org/t/p/w500/' + data['poster_path']
    else:
        return 'https://westsiderc.org/wp-content/uploads/2019/08/Image-Not-Available.png'


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = list()
    recommended_movies_posters = list()
    for i in distances[1:7]:
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movies.iloc[i[0]].movie_id))
    return recommended_movies, recommended_movies_posters,


st.title('Movie Recommender System')

selected_movie = st.selectbox(
    'Select a Movie',
    movies['title'].values, key='selected')

if 'selected' in st.session_state:
    idx = movies[movies['title'] == selected_movie]['movie_id'].values[0]
    link = fetch_poster(idx)
    col001, col002, col003 = st.columns(3)
    with col002:
        st.image(link, width=250)

col01, col02, col03, col04, col05, col06, col07 = st.columns(7)
if col04.button('Recommend'):
    names, posters = recommend(selected_movie)
    col11, col22, col33 = st.columns(3)
    with col11:
        st.image(posters[0], width=200)
        st.markdown("<h6 style='text-align: center; overflow:visible; color: white;'>{0}</h6>".format(names[0]),
                    unsafe_allow_html=True)
    with col22:
        st.image(posters[1], width=200)
        st.markdown("<h6 style='text-align: center; overflow:visible; color: white;'>{0}</h6>".format(names[1]),
                    unsafe_allow_html=True)
    with col33:
        st.image(posters[2], width=200)
        st.markdown("<h6 style='text-align: center; overflow:visible; color: white;'>{0}</h6>".format(names[2]),
                    unsafe_allow_html=True)

    col44, col55, col66 = st.columns(3)
    with col44:
        st.image(posters[3], width=200)
        st.markdown("<h6 style='text-align: center; overflow:visible; color: white;'>{0}</h6>".format(names[3]),
                    unsafe_allow_html=True)
    with col55:
        st.image(posters[4], width=200)
        st.markdown("<h6 style='text-align: center; overflow:visible; color: white;'>{0}</h6>".format(names[4]),
                    unsafe_allow_html=True)
    with col66:
        st.image(posters[5], width=200)
        st.markdown("<h6 style='text-align: center; overflow:visible; color: white;'>{0}</h6>".format(names[5]),
                    unsafe_allow_html=True)
