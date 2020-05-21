import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

movies_df = pd.read_csv(r'Week5\Content-based Recommender\movies.csv')
ratings_df = pd.read_csv(r'Week5\Content-based Recommender\ratings.csv')
# print(movies_df.head())
# print(ratings_df.head())

# Making the data easier to analyse
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df['genres'] = movies_df.genres.str.split('|')
ratings_df = ratings_df.drop('timestamp', 1)

moviesWithGenres_df = movies_df.copy()
#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
# print(moviesWithGenres_df.head())

# Collecting User input
userInput = [{
    'title': 'Breakfast Club, The',
    'rating': 5
}, {
    'title': 'Toy Story',
    'rating': 3.5
}, {
    'title': 'Jumanji',
    'rating': 2
}, {
    'title': "Pulp Fiction",
    'rating': 5
}, {
    'title': 'Akira',
    'rating': 4.5
}]
inputMovies = pd.DataFrame(userInput)
# Filtering out movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
# print(inputMovies)

#Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(
    inputMovies['movieId'].tolist())]
userMovies = userMovies.reset_index(drop=True)
#Dropping unnecessary columns to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title',
                                                    1).drop('genres',
                                                            1).drop('year', 1)

#The user profile
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
# print(userProfile)

#Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
#And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title',
                                                1).drop('genres',
                                                        1).drop('year', 1)

#Multiply the genres by the weights and then take the weighted average
recommendationTable_df = (
    (genreTable * userProfile).sum(axis=1)) / (userProfile.sum())
recommendationTable_df = recommendationTable_df.sort_index(ascending=True)
# print(recommendationTable_df.head())
#Just a peek at the values

movies_df = movies_df.loc[movies_df['movieId'].isin(
    recommendationTable_df.keys())]
movies_df['rec'] = recommendationTable_df.values
movies_df = movies_df.sort_values('rec', ascending=False)
movies_df = movies_df.reset_index(drop=True)
movies_df.to_csv(r'Week5\Content-based Recommender\movie_recommendations.csv')
print(movies_df.head(20))
