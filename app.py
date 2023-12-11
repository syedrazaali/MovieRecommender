import dash
from dash import html, dcc, callback, Input, Output, State
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import dash_bootstrap_components as dbc
import os


def get_movie_image_url(movie_id):
    return f"https://liangfgithub.github.io/MovieImages/{movie_id}.jpg?raw=true"


# Load movies and ratings data
movies_df = pd.read_csv('https://razaalimovierecommendation.s3.amazonaws.com/movies.dat', sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
ratings_df = pd.read_csv('https://razaalimovierecommendation.s3.amazonaws.com/ratings.dat', sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin-1')

movie_id_mapping = {row['Title']: row['MovieID'] for index, row in movies_df.iterrows()}

# Calculate the global average rating and the global number of ratings
global_average_rating = ratings_df['Rating'].mean()
global_rating_count = ratings_df['Rating'].count()

# Set a minimum threshold of ratings needed to be considered
min_ratings_threshold = ratings_df.groupby('MovieID')['Rating'].count().quantile(0.8)

# Compute the Bayesian average for each movie
def bayesian_average(row, C=min_ratings_threshold, m=global_average_rating):
    v = row['RatingCount']
    R = row['AverageRating']
    return (v / (v + C) * R) + (C / (C + v) * m)

# Group the ratings by movie and calculate the average rating and rating count
movie_stats = ratings_df.groupby('MovieID').agg(AverageRating=('Rating', 'mean'), RatingCount=('Rating', 'count')).reset_index()

# Merge the movie stats with the movies dataframe
movies_with_stats = pd.merge(movies_df, movie_stats, on='MovieID', how='left')

# Apply the bayesian_average function to each row in the movies_with_stats dataframe
movies_with_stats['BayesianAverage'] = movies_with_stats.apply(bayesian_average, axis=1)

# Sort the movies based on the BayesianAverage
sorted_movies = movies_with_stats.sort_values('BayesianAverage', ascending=False)

# Normalize the rating matrix by centering each row
user_rating_avg = ratings_df.groupby('UserID')['Rating'].mean()
ratings_df_centered = ratings_df.join(user_rating_avg, on='UserID', rsuffix='_mean')
ratings_df_centered['Rating_Centered'] = ratings_df_centered['Rating'] - ratings_df_centered['Rating_mean']

# Ensure MovieID is treated consistently as strings
ratings_df_centered['MovieID'] = ratings_df_centered['MovieID'].astype(str)

# Create the rating matrix without filling NaNs
rating_matrix = ratings_df_centered.pivot(index='UserID', columns='MovieID', values='Rating_Centered')

# Fill NaNs with 0 for the purpose of cosine similarity calculation
rating_matrix_for_similarity = rating_matrix.fillna(0)

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(rating_matrix_for_similarity.T)
cosine_sim_df = pd.DataFrame(cosine_sim, index=rating_matrix_for_similarity.columns.astype(str), columns=rating_matrix_for_similarity.columns.astype(str))

# Calculate common ratings count to avoid dimension misalignment
bool_rating_matrix = rating_matrix.notnull().astype(int)
common_ratings_count = bool_rating_matrix.T @ bool_rating_matrix

# Apply similarity threshold
cosine_sim_df_masked = cosine_sim_df.where(common_ratings_count >= 2, np.nan)  # Lowered the threshold to 2

# Read the similarity matrix from S3 bucket
similarity_matrix_url = 'https://razaalimovierecommendation.s3.amazonaws.com/similarity_matrix.csv'
cosine_sim_df_masked = pd.read_csv(similarity_matrix_url, index_col=0)
cosine_sim_df_masked.columns = cosine_sim_df_masked.columns.astype(str)
cosine_sim_df_masked.index = cosine_sim_df_masked.index.astype(str)

# Construct the similarity matrix
N = 30
top_n_similarities = cosine_sim_df_masked.apply(lambda x: x.sort_values(ascending=False).iloc[:N], axis=1)

# System I: Genre-Based Recommendation Function
def recommend_movies_genre(genre, top_n=10):
    genre_movies = sorted_movies[sorted_movies['Genres'].str.contains(genre)]
    return genre_movies[['Title', 'BayesianAverage']].head(top_n)

# System II: Collaborative Filtering Recommendation Function
def myIBCF(new_user_ratings, similarity_matrix, top_n_similarities, rating_matrix, top_n=10):
    new_user_ratings_series = pd.Series(new_user_ratings, dtype='float64')
    predictions = {}
    for movie in similarity_matrix.columns:
        if str(movie) not in new_user_ratings:
            similar_movies = top_n_similarities.loc[str(movie)].dropna().index
            similar_movies = similar_movies.intersection(new_user_ratings_series.index)
            if not similar_movies.empty:
                sim_scores = similarity_matrix.loc[str(movie), similar_movies]
                movie_ratings = new_user_ratings_series.loc[similar_movies]
                predicted_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
                if predicted_rating > 0:
                    predictions[str(movie)] = predicted_rating
    predictions_series = pd.Series(predictions).sort_values(ascending=False).head(top_n)
    if predictions_series.empty:
        # Fallback strategy
        user_unrated_movies = rating_matrix.columns.difference(new_user_ratings_series.index)
        top_movies = rating_matrix.loc[:, user_unrated_movies].mean().sort_values(ascending=False).head(top_n).index.tolist()
        return top_movies
    return predictions_series.index.tolist()

# Set up the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Define a list of sample movies for users to rate in System II
sample_movies = sorted_movies['Title'].head(120).tolist()

# Main app layout
app.layout = html.Div([
    html.H1("Movie Recommender System"),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='System I: Genre-Based', value='tab-1'),
        dcc.Tab(label='System II: Collaborative Filtering', value='tab-2'),
    ]),
    html.Div(id='tabs-content'),

    # Separately place the submit button outside the tabs-content
    html.Button('Submit Ratings', id='submit-ratings', n_clicks=0, style={
        'display': 'block',
        'margin-top': '20px',  # Adjust this value to move the button further down
        'margin-left': 'auto',
        'margin-right': 'auto'
    })
])

# Layout component for System II: User Rating Input
def create_rating_input(movie_titles):
    movie_cards = []
    for i, movie in enumerate(movie_titles):
        movie_card = html.Div(
            [
                html.Img(
                    src=get_movie_image_url(movie_id_mapping[movie]),
                    style={
                        'width': '150px',  # Adjust width as needed
                        'height': 'auto',  # Adjust height as needed
                        'margin': '10px'
                    }
                ),
                dcc.RadioItems(
                    id={'type': 'user-rating', 'index': i},
                    options=[{'label': str(j), 'value': j} for j in range(1, 6)],
                    value=None,
                    labelStyle={'display': 'inline-block', 'margin-right': '5px'}, # Adjust the right margin to bring radio buttons closer
                    inputStyle={"margin-right": "5px"}, # Adjust space between radio button and label
                    style={'display': 'flex', 'justifyContent': 'center'}  # Align the radio buttons in a row centered
                )
            ],
            style={
                'display': 'inline-block',  # This makes it align in a grid
                'width': '200px',  # Adjust the width as needed
                'verticalAlign': 'top'  # Align the tops of the movie cards
            }
        )
        movie_cards.append(movie_card)

    return html.Div(
        movie_cards,
        style={
            'display': 'flex',  # Use flexbox to create a shelf-like layout
            'flexWrap': 'wrap',  # Allow the cards to wrap to the next line
            'justifyContent': 'space-around',  # Evenly space the movie cards
            'overflowY': 'auto',  # Add a scrollbar if necessary
            'alignItems': 'flex-start',  # Align items at the top
            'height': '500px'  # Adjust the height of the shelf
        }
    )


# Callback for rendering tabs content
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Select your favorite genre'),
            dcc.Dropdown(
                id='genre-dropdown',
                options=[{'label': genre, 'value': genre} for genre in ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western' ]],
                value='Comedy'
            ),
            html.Button('Get Recommendations', id='get-recommendations', n_clicks=0),
            html.Div(id='genre-recommendations-output')
        ])
    elif tab == 'tab-2':
        return html.Div([
            create_rating_input(sample_movies),
            html.Button('Submit Ratings', id='submit-ratings', n_clicks=0),
            html.Div(id='cf-recommendations-output')
        ])

# Callback for genre-based recommendations
@app.callback(Output('genre-recommendations-output', 'children'),
              Input('get-recommendations', 'n_clicks'),
              State('genre-dropdown', 'value'))
def update_genre_recommendations(n_clicks, selected_genre):
    if n_clicks > 0:
        recommendations = recommend_movies_genre(selected_genre)
        return html.Ul([html.Li(f"{row['Title']} - Bayesian Average: {row['BayesianAverage']:.2f}") for index, row in recommendations.iterrows()])
    return html.Div()

# Callback for collaborative filtering recommendations
@app.callback(Output('cf-recommendations-output', 'children'),
              Input('submit-ratings', 'n_clicks'),
              [State({'type': 'user-rating', 'index': i}, 'value') for i in range(len(sample_movies))])
def update_cf_recommendations(n_clicks, *ratings):
    if n_clicks > 0:
        user_ratings = {str(movie_id_mapping[movie]): rating for movie, rating in zip(sample_movies, ratings) if rating is not None}
        recommended_movie_ids = myIBCF(user_ratings, cosine_sim_df_masked, top_n_similarities, rating_matrix)
        recommended_movies = movies_df[movies_df['MovieID'].isin(recommended_movie_ids)]
        return [html.P(movies_df.loc[movies_df['MovieID'] == int(movie_id), 'Title'].iloc[0])
                for movie_id in recommended_movie_ids]
    return html.Div()

if __name__ == '__main__':
    # Get the port to listen on (default to 8050 if not specified)
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)
