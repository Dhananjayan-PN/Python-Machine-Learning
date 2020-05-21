import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

movies_df = pd.read_csv(
    r'Recommender Systems\Collaborative Filtering Recommender\movies.csv')
ratings_df = pd.read_csv(
    r'Recommender Systems\Collaborative Filtering Recommender\ratings.csv')

# Making the data easier to analyse
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df['genres'] = movies_df.genres.str.split('|')
ratings_df = ratings_df.drop('timestamp', 1)
movies_df = movies_df.drop('genres', 1)
# print(movies_df.head())
# print(ratings_df.head())

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

#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)

#Filtering out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(
    inputMovies['movieId'].tolist())]
userSubsetGroup = userSubset.groupby(['userId'])
#Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,
                         key=lambda x: len(x[1]),
                         reverse=True)
userSubsetGroup = userSubsetGroup[0:100]

pearsonCorrelationDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)
    #Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(
        group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList
               ]) - pow(sum(tempRatingList), 2) / float(nRatings)
    Syy = sum([i**2 for i in tempGroupList
               ]) - pow(sum(tempGroupList), 2) / float(nRatings)
    Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)
              ) - sum(tempRatingList) * sum(tempGroupList) / float(nRatings)

    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
    else:
        pearsonCorrelationDict[name] = 0

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))

topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsersRating = topUsers.merge(ratings_df,
                                left_on='userId',
                                right_on='userId',
                                how='inner')
topUsersRating['weightedRating'] = topUsersRating[
    'similarityIndex'] * topUsersRating['rating']
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[[
    'similarityIndex', 'weightedRating'
]]
tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']
#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df[
    'weighted average recommendation score'] = tempTopUsersRating[
        'sum_weightedRating'] / tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df = recommendation_df.sort_values(by='movieId', ascending=True)
print(recommendation_df.head(10))

movies_df = movies_df.loc[movies_df['movieId'].isin(
    recommendation_df.head(10)['movieId'].tolist())]
movies_df['rec'] = recommendation_df['weighted average recommendation score']
movies_df = movies_df.sort_values(by='rec', ascending=False)
movies_df = movies_df.reset_index(drop=True)
movies_df.to_csv(
    r'Recommender Systems\Content-based Recommender\movie_recommendations.csv')
print(movies_df.head(20))
