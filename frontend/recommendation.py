import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import pickle
import random

title_length, title_set, genres2int, features, target_values, ratings, users, \
movies, data, movies_orig, users_orig = pickle.load(open('./data/params.p', mode='rb'))
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}
sentences_size = title_length
load_dir = './save_model/'
movie_feature_size = user_feature_size = 512
movie_matrix_path = './data/movie_matrix.p'
user_matrix_path = './data/user_matrix.p'


def get_tensors(loaded_graph):
	"""
	The function "get_tensors" returns the tensors needed for making predictions in a loaded TensorFlow
	graph.
	"""
	uid = loaded_graph.get_tensor_by_name("uid:0")
	user_gender = loaded_graph.get_tensor_by_name("user_gender:0")
	user_age = loaded_graph.get_tensor_by_name("user_age:0")
	user_job = loaded_graph.get_tensor_by_name("user_job:0")
	movie_id = loaded_graph.get_tensor_by_name("movie_id:0")
	movie_categories = loaded_graph.get_tensor_by_name("movie_categories:0")
	movie_titles = loaded_graph.get_tensor_by_name("movie_titles:0")
	targets = loaded_graph.get_tensor_by_name("targets:0")
	dropout_keep_prob = loaded_graph.get_tensor_by_name("dropout_keep_prob:0")
# Two different methods for calculating predicted ratings use different names to obtain the tensor inference.
	inference = loaded_graph.get_tensor_by_name(
		"inference/ExpandDims:0")  
	movie_combine_layer_flat = loaded_graph.get_tensor_by_name("movie_fc/Reshape:0")
	user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc/Reshape:0")
	return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat


# Predicting the rating given by a specific user to a specific movie
# This part involves performing forward propagation on the network to compute the predicted rating.
def rating_movie(user_id_val, movie_id_val):
	loaded_graph = tf.Graph()  #
	with tf.compat.v1.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		# Get Tensors from loaded model
		uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat = get_tensors(loaded_graph)  # loaded_graph

		categories = np.zeros([1, 18])
		categories[0] = movies.values[movieid2idx[movie_id_val]][2]

		titles = np.zeros([1, sentences_size])
		titles[0] = movies.values[movieid2idx[movie_id_val]][1]

		feed = {
			uid: np.reshape(users.values[user_id_val - 1][0], [1, 1]),
			user_gender: np.reshape(users.values[user_id_val - 1][1], [1, 1]),
			user_age: np.reshape(users.values[user_id_val - 1][2], [1, 1]),
			user_job: np.reshape(users.values[user_id_val - 1][3], [1, 1]),
			movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
			movie_categories: categories,  # x.take(6,1)
			movie_titles: titles,  # x.take(5,1)
			dropout_keep_prob: 1}

		# Get Prediction
		inference_val = sess.run([inference], feed)

		return inference_val

# Generating movie feature matrix
# Combine the trained movie features to create the movie feature matrix and save it locally.
# Perform forward propagation for each movie.
def save_movie_feature_matrix():
	loaded_graph = tf.Graph()  #
	movie_matrics = []
	with tf.compat.v1.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		# Get Tensors from loaded model
		uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat = get_tensors(loaded_graph)  # loaded_graph

		for item in movies.values:
			categories = np.zeros([1, 18])
			categories[0] = item.take(2)

			titles = np.zeros([1, sentences_size])
			titles[0] = item.take(1)

			feed = {
				movie_id: np.reshape(item.take(0), [1, 1]),
				movie_categories: categories,  # x.take(6,1)
				movie_titles: titles,  # x.take(5,1)
				dropout_keep_prob: 1}

			movie_combine_layer_flat_val = sess.run([movie_combine_layer_flat], feed)
			movie_matrics.append(movie_combine_layer_flat_val)

	pickle.dump((np.array(movie_matrics).reshape(-1, 200)), open(movie_matrix_path, 'wb'))


# Generating user feature matrix
# Combine the trained user features to create the user feature matrix and save it locally.
# Perform forward propagation for each user.
def save_user_feature_matrix():
	loaded_graph = tf.Graph()  #
	users_matrics = []
	with tf.compat.v1.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		# Get Tensors from loaded model
		uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat = get_tensors(loaded_graph)  # loaded_graph

		for item in users.values:
			feed = {
				uid: np.reshape(item.take(0), [1, 1]),
				user_gender: np.reshape(item.take(1), [1, 1]),
				user_age: np.reshape(item.take(2), [1, 1]),
				user_job: np.reshape(item.take(3), [1, 1]),
				dropout_keep_prob: 1}

			user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)
			users_matrics.append(user_combine_layer_flat_val)

	pickle.dump((np.array(users_matrics).reshape(-1, 200)), open(user_matrix_path, 'wb'))


def load_feature_matrix(path):
	if os.path.exists(path):
		pass
	else:
		if path == movie_matrix_path:
			save_movie_feature_matrix()
		else:
			save_user_feature_matrix()
	return pickle.load(open(path, 'rb'))




# Recommending movies of the same genre using the movie feature matrix
# The approach is to calculate the cosine similarity between the feature vector of the specified movie and the entire movie feature matrix.
# Select the top_k movies with the highest similarity.
# ToDo: Incorporate random selection to ensure slightly different recommendations each time.
def recommend_same_type_movie(movie_id_val, top_k=20):
	"""
	The function `recommend_same_type_movie` takes a movie ID as input and recommends similar movies
	based on a pre-trained model and a movie feature matrix.
	"""
	movie_matrics = load_feature_matrix(movie_matrix_path)
	loaded_graph = tf.Graph()  #
	print("here ----------- recommending same type of mvies")
	with tf.compat.v1.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)
		norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keep_dims=True))
		normalized_movie_matrics = movie_matrics / norm_movie_matrics
		probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
		probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
		sim = (probs_similarity.eval())
		p = np.squeeze(sim)
		p[np.argsort(p)[:-top_k]] = 0
		p = p / np.sum(p)
		results = set()
		while len(results) != 5:
			c = np.random.choice(3883, 1, p=p)[0]
			results.add(c)
		recom_movies = []
		movie_you_watched = list(movies_orig[movieid2idx[movie_id_val]])
		for val in results:
			# print(val)
			print(movies_orig[val])
			recom_movies.append(list(movies_orig[val]))

		# return results
		return movie_you_watched, recom_movies



# Recommending movies that a specific user might like
# The approach is to calculate the ratings for all movies using the user feature vector and movie feature matrix.
# Select the top_k movies with the highest ratings.
# ToDo: Incorporate random selection.
def recommend_your_favorite_movie(user_id_val, top_k=20):
	"""
	The function `recommend_your_favorite_movie` takes a user ID as input and returns information about
	the user and a list of recommended movies based on a trained model.
	:return: the user information and a list of recommended movies.
	"""
	user_matrics = load_feature_matrix(user_matrix_path)
	movie_matrics = load_feature_matrix(movie_matrix_path)
	loaded_graph = tf.Graph()  #
	with tf.compat.v1.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)
		probs_embeddings = (user_matrics[user_id_val - 1]).reshape([1, 200])
		probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
		sim = (probs_similarity.eval())
		p = np.squeeze(sim)
		p[np.argsort(p)[:-top_k]] = 0
		p = p / np.sum(p)
		results = set()
		while len(results) != 5:
			c = np.random.choice(3883, 1, p=p)[0]
			results.add(c)
		recom_movies = []
		for val in results:
			recom_movies.append(list(movies_orig[val]))
		your_info = users_orig[user_id_val - 1]
		return your_info, recom_movies


# Movies that people who have watched this movie might also (like) other movies
# First, select the top_k users who liked a particular movie and obtain their user feature vectors.
# Then, calculate the ratings of these users for all movies.
# Choose the movies with the highest ratings from each user as recommendations.
# ToDo: Incorporate random selection.
def recommend_other_favorite_movie(movie_id_val, top_k=20):
	print('got here -------------- recommending')
	user_matrics = load_feature_matrix(user_matrix_path)
	movie_matrics = load_feature_matrix(movie_matrix_path)
	loaded_graph = tf.Graph()  #
	with tf.compat.v1.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
		probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(user_matrics))
		favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0][-top_k:]
		users_info = users_orig[favorite_user_id - 1]
		probs_users_embeddings = (user_matrics[favorite_user_id - 1]).reshape([-1, 200])
		probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
		sim = (probs_similarity.eval())
		p = np.argmax(sim, 1)

		results = set()
		while len(results) != 5:
			c = p[random.randrange(top_k)]
			results.add(c)
		recom_movies = []
		movie_you_watched = list(movies_orig[movieid2idx[movie_id_val]])
		for val in (results):
			recom_movies.append(list(movies_orig[val]))

		# return results
		return movie_you_watched, recom_movies, users_info


# Print recommendations for your favorite movie
# print(recommend_your_favorite_movie(222))

# # Predict the rating given by a specific user to a specific movie
# prediction_rating = rating_movie(user_id_val=123, movie_id_val=1234)
# print('For user: 123, predicting the rating for movie: 1234', prediction_rating)

# # Generate feature matrices for users and movies and store them locally
# save_movie_feature_matrix()
# save_user_feature_matrix()

# # Recommend other top k movies of the same genre for a given movie
# results = recommend_same_type_movie(movie_id_val=666, top_k=5)
# print(results)

# # Recommend top k movies that a given user might like
# results = recommend_your_favorite_movie(user_id_val=222, top_k=20)
# print(results)

# Recommend other favorite movies that people who watched this movie might like
# print(recommend_other_favorite_movie(movie_id_val=666, top_k=5))

# Variables and datasets description
# title_length: Length of the Title field (15)
# title_set: Set of Title text
# genres2int: Dictionary to convert movie genres to numbers
# features: Input X
# targets_values: Learning targets y
# ratings: Pandas object of the rating dataset
# users: Pandas object of the user dataset
# movies: Pandas object of the movie dataset
# data: Pandas object combining three datasets together
# movies_orig: Original raw movie dataset without data processing
# users_orig: Original raw user dataset without data processing
