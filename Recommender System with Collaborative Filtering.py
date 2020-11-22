# Import necessary modules

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import pandas as pd
import random
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise import SVDpp
from surprise.model_selection import train_test_split

#Import train.txt file and create matrix dataframe
print("Importing train.txt file and creating matrix...")

matrix = pd.read_csv ('train.txt', header = None, sep=' ')
matrix.columns = ['userid', 'movieid', 'rating']

users = matrix.userid.unique()
movies = matrix.movieid.unique()
movies.sort()
users.sort()

# Get the list of unrated movies

unrated_movies = []
for i in range(1,movies[-1]+1):
    not_in_list = i not in movies
    if not_in_list == True:
        unrated_movies.append(i)

user_item_matrix = matrix.pivot_table(index= 'userid', columns= 'movieid', values = "rating")

# Add unrated movies to the matrix

rows = []
for i in unrated_movies:
    rand_user = random.randint(1,users[-1]+1)
    new_row = {'userid':np.int(rand_user) , 'movieid':np.int(i) , 'rating':np.int(user_item_matrix.iloc[rand_user].mean(axis=0))}
    rows.append(new_row)
matrix = matrix.append(rows, ignore_index=True)

user_item_matrix = matrix.pivot_table(index= 'userid', columns= 'movieid', values = "rating")

print("Matrix is created.")

# Creating the Train and Test Sets

reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(matrix[['userid', 'movieid', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=.10)

print("Dataset is created.")


print("SVD++ Algorithm is running...")

# Running the SVD++ estimation algorithm

algo = SVDpp()
algo.fit(trainset)


print("Running the trained model against the testset...")


test_pred = algo.test(testset)


print("SVD++ : Test Set")
accuracy.rmse(test_pred, verbose=True)


print("SVD++ : Training Set")
train_pred = algo.test(trainset.build_testset())
accuracy.rmse(train_pred)


print("Creating the output file...")
users = matrix.userid.unique()
movies = matrix.movieid.unique()
movies.sort()
users.sort()

# Getting estimations and creating the output file according to the guidelines

my_recs = []
for uid in users:
    user_rats = np.array(user_item_matrix.iloc[uid-1])
    for iid in movies:
        rating = user_rats[iid-1] 
        if rating not in range(1,6): # Check if user is already rated the movie

            estimation = algo.predict(uid=uid,iid=iid).est

            if isinstance(estimation, int):
                my_recs.append((uid, iid, estimation))
            else:
                my_recs.append((uid, iid, estimation.round()))
        else:
            my_recs.append((uid, iid, rating))

output = pd.DataFrame(my_recs, columns=['uid','iid', 'predictions']).sort_values(['uid', 'iid'], ascending=True)
np.savetxt(r'submit_sample.txt', output.values, fmt='%d')

print("submit_sample.txt created.")
