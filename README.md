# Movie Recommender System

This application is a movie recommendation system built with Dash, a Python framework for building analytical web applications. The system uses a dataset of movies and user ratings to provide two types of recommendations:

- **System I: Genre-Based** - Recommends top movies within a selected genre.
- **System II: Collaborative Filtering** - Provides recommendations based on user ratings.

## Features

- Users can rate a sample set of movies.
- Users can select their favorite genre to get recommendations.
- Movie ratings are used to calculate recommendations using collaborative filtering.

## Data

The application uses two main datasets:

- `movies.dat`: Contains movie titles and genres.
- `ratings.dat`: Includes user ratings for different movies.

These datasets are hosted on an S3 bucket and are loaded into the application to perform the recommendation logic.

## Bayesian Average Rating

To account for movies with very few ratings, the application uses a Bayesian average for the movie scoring system.

## Cosine Similarity

The collaborative filtering mechanism is based on the cosine similarity of user ratings across movies, allowing the system to find and recommend movies that are similar to the user's tastes.

## Running the Application

To run the application, use the following command:

```bash
python app.py
