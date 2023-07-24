import tensorflow as tf
import pickle

features = pickle.load(open('./data/features.p', mode='rb'))
# Dimension of the embedding matrix
embed_dim = 32
# The reason for adding +1 below is because the IDs and the actual counts differ by 1
# Number of users
uid_max = max(features.take(0, 1)) + 1  # 6040
# Number of genders
gender_max = max(features.take(2, 1)) + 1  # 1 + 1 = 2
# Number of age categories
age_max = max(features.take(3, 1)) + 1  # 6 + 1 = 7
# Number of job categories
job_max = max(features.take(4, 1)) + 1  # 20 + 1 = 21

def get_inputs():
    uid = tf.placeholder(tf.int32, [None, 1], name="uid")
    user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
    user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
    user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")
    return uid, user_gender, user_age, user_job

def get_user_embedding(uid, user_gender, user_age, user_job):
    with tf.name_scope("user_embedding"):
        # Operations below are similar to converting words to word embeddings in sentiment analysis projects
        # User feature dimension set to 32
        # Initialize a large user matrix first
        # tf.random_uniform's second parameter is the minimum value for initialization (here, -1),
        # and the third parameter is the maximum value (here, 1)
        uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max, embed_dim], -1, 1), name="uid_embed_matrix")
        # Find the corresponding embedding layer for the specified user ID
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name="uid_embed_layer")

        # Gender feature dimension set to 32
        gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max, embed_dim], -1, 1), name="gender_embed_matrix")
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name="gender_embed_layer")

        # Age feature dimension set to 32
        age_embed_matrix = tf.Variable(tf.random_uniform([age_max, embed_dim], -1, 1), name="age_embed_matrix")
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")

        # Job feature dimension set to 32
        job_embed_matrix = tf.Variable(tf.random_uniform([job_max, embed_dim], -1, 1), name="job_embed_matrix")
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name="job_embed_layer")
    # Return the generated user data
    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer

def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
    with tf.name_scope("user_fc"):
        # First fully connected layer
        # tf.layers.dense's first parameter is the input, and the second parameter is the number of units in the layer
        uid_fc_layer = tf.layers.dense(uid_embed_layer, embed_dim, name="uid_fc_layer", activation=tf.nn.relu)
        gender_fc_layer = tf.layers.dense(gender_embed_layer, embed_dim, name="gender_fc_layer", activation=tf.nn.relu)
        age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim, name="age_fc_layer", activation=tf.nn.relu)
        job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, name="job_fc_layer", activation=tf.nn.relu)

        # Second fully connected layer
        # Combine the above segments into a complete fully connected layer
        user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)  # (?, 1, 128)
        # Verify if the generated tensorflow is of dimension 128
        # tf.contrib.layers.fully_connected's first parameter is the input, and the second parameter is the output
        # Here, the input is user_combine_layer and the output is 200, which means each user has 200 features
        # It's like a 200-class problem, where the output will represent the likelihood of each feature
        user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)  # (?, 1, 200)

        user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])
    return user_combine_layer, user_combine_layer_flat
