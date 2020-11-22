# Recommender System with Collaborative Filtering

In this project, I built a recommender system with collaborative filtering based on users' previous ratings.

  - Python's Surprise module implements Singular Value Decomposition (SVD++) by taking implicit ratings into account and I used this module in order to predict the user ratings on unrated movies.
  - The famous SVD algorithm, as popularized by Simon Funk during the Netflix Prize. When baselines are not used, this is equivalent to Probabilistic Matrix Factorization.
  - The SVD++ algorithm, an extension of SVD taking into account implicit ratings.

Steps involved:
  1) Import necessary modules and train.txt file.
  2) Create ratings matrix.
  3) Get the list of unrated movies and add them to the ratings matrix.
  4) Prepate the dataset with Reader.
  5) Split train and test sets.
  6) Run the SVD++ algorithm on trainset.
  7) Create the sample_output.txt file with the format according to the project guidelines.
  
Instructions on running the recommender system:
  1) Software is tested and built on Python 3.9.0 Shell
  2) Necessary modules: Random, Pandas, Surprise
  3) Open Python 3.9.0 Shell and run "Recommender System with Collaborative Filtering.py"
  
References:
1) : https://kerpanic.wordpress.com/2018/03/26/a-gentle-guide-to-recommender-systems-with-surprise/

2) : https://surprise.readthedocs.io/en/stable/matrix_factorization.html

3) : https://pandas.pydata.org/pandas-docs/stable/reference/api/

4) : https://towardsdatascience.com/user-user-collaborative-filtering-for-jokes-recommendation-b6b1e4ec8642

5) : https://realpython.com/build-recommendation-engine-collaborative-filtering/

6) : https://medium.com/@tomar.ankur287/user-user-collaborative-filtering-recommender-system-51f568489727
