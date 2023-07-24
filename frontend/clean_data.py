import pandas as pd
import os
import pickle


def process_user_data():
	print('processing users data....')
	# read users data
	users_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
	users = pd.read_table('movielens/ml-1m/users.dat', sep='::', header=None, names=users_title, engine='python')
	users = users.filter(regex='UserID|Gender|Age|OccupationID')
	users_original = users.values
	gender_map = {'F': 0, 'M': 1}
	users['Gender'] = users['Gender'].map(gender_map)
	age_map = {val: ii for ii, val in enumerate(sorted(list(set(users['Age']))))}
	users['Age'] = users['Age'].map(age_map)
	return users, users_original


def process_movie_data():
	print('processing movies data....')
	# read movies data
	movies_title = ['MovieID', 'Title', 'Genres']
	# movies = pd.read_table('movielens/ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python')
	movies = pd.read_table('movielens/ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python', encoding='latin-1')

	movies_original = movies.values

	movies['Title'] = movies['Title'].str.split('(').str[0].str.strip()
	title_to_id_df = movies[['Title', 'MovieID']]

	# 电影Title转数字字典
	title_set = set()
	for val in movies['Title'].str.split():
		title_set.update(val)

	title_set.add('PADDING')
	title2int = {val: ii for ii, val in enumerate(title_set)}

	# 将电影Title转成等长数字列表，长度是15
	title_length = 15
	title_map = {val: [title2int[row] for row in val.split()] for val in set(movies['Title'])}

	for key in title_map.keys():
		padding_length = title_length - len(title_map[key])
		padding = [title2int['PADDING']] * padding_length
		title_map[key].extend(padding)

	movies['Title'] = movies['Title'].map(title_map)

	# 电影类型转数字字典
	genres_set = set()
	for val in movies['Genres'].str.split('|'):
		genres_set.update(val)

	genres_set.add('PADDING')
	genres2int = {val: ii for ii, val in enumerate(genres_set)}

	# 将电影类型转成等长数字列表
	genres_map = {val: [genres2int[row] for row in val.split('|')] for val in set(movies['Genres'])}

	# 将每个样本的电影类型数字列表处理成相同长度，长度不够用'PADDING'填充
	for key in genres_map:
		for cnt in range(max(genres2int.values()) - len(genres_map[key])):
			genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['PADDING'])

	movies['Genres'] = movies['Genres'].map(genres_map)
	return movies, movies_original, genres2int, title_set, title_length, title_to_id_df


def process_rating_data():
	print('processing ratings data....')
	# read rating data
	ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
	ratings = pd.read_table('movielens/ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
	ratings = ratings.filter(regex='UserID|MovieID|ratings')
	return ratings


def save_data():
	users, users_original = process_user_data()
	movies, movies_original, genres2int, title_set, title_length, title_to_id_df = process_movie_data()
	ratings = process_rating_data()
	data = pd.merge(pd.merge(ratings, users), movies)

	target_fields = ['ratings']
	feature_pd, tragets_pd = data.drop(target_fields, axis=1), data[target_fields]

	features = feature_pd.values
	targets = tragets_pd.values

	data_path = './data'
	if not os.path.exists(data_path):
		os.makedirs(data_path)
	with open(data_path + '/features.p', 'wb') as file:
		pickle.dump(features, file)
	with open(data_path + '/targets.p', 'wb') as file:
		pickle.dump(targets, file)
	params = (title_length, title_set, genres2int, features, targets, ratings, users, movies, data, movies_original, users_original)
	with open(data_path + '/params.p', 'wb') as file:
		pickle.dump(params, file)

	title_vocb_num = len(title_set) #5216
	genres_num = len(genres2int) #19
	movie_id_num = max(movies['MovieID']) + 1 #3952
	argument = (title_vocb_num, genres_num, movie_id_num)
	with open(data_path + '/argument.p', 'wb') as file:
		pickle.dump(argument, file)

	title_to_id_df.to_csv(data_path + '/title_to_id.csv', sep=',', encoding='utf-8', index=False)

	print('Data saved.')


save_data()
