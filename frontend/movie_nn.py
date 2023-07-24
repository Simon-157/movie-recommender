import tensorflow as tf
import pickle

# Feature information: ['UserID' 'MovieID' 'Gender' 'Age' 'JobID' 'Title' 'Genres']
title_length, title_set, genres2int, features, target_values, ratings, users, \
movies, data, movies_orig, users_orig = pickle.load(open('./data/params.p', 'rb'))

title_vocb_num, genres_num, movie_id_num = pickle.load(open('./data/argument.p', 'rb'))

embed_dim = 32

# Number of movie IDs
movie_id_max = movie_id_num
# Number of movie categories, including 'PADDING'
movie_categories_max = genres_num
# Number of words in movie titles
movie_title_max = title_vocb_num

# Movie title length for word embedding; set to 15 for fixed dimensions
# If the title is shorter, it will be padded with blank characters; if longer, it will be truncated
sentences_size = title_length
# Text convolution sliding windows with sizes of 2, 3, 4, 5 words
window_sizes = {2, 3, 4, 5}
# Number of text convolution kernels
filter_num = 8

# Dictionary to map movie IDs to indices; in the dataset, movie IDs may not match their indices
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}


def get_inputs():
    """
    Get input tensors for all movie features
    """
    movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
    # Movie categories need to remove 'PADDING', so the size is reduced by 1
    movie_categories = tf.placeholder(tf.int32, [None, movie_categories_max - 1], name="movie_categories")
    movie_titles = tf.placeholder(tf.int32, [None, 15], name="movie_titles")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    return movie_id, movie_categories, movie_titles, dropout_keep_prob


def get_movie_id_embed_layer(movie_id):
    """
    Get the embedding layer for movie IDs
    """
    with tf.name_scope("movie_embedding"):
        movie_id_embed_matrix = tf.Variable(tf.random_uniform([movie_id_max, embed_dim], -1, 1), name="movie_id_embed_matrix")
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name="movie_id_embed_layer")
    return movie_id_embed_layer


# Sum multiple embedding vectors of movie categories
def get_movie_categories_embed_layer(movie_categories, combiner='sum'):
    """
    Define the embedding for movie categories and combine all types of a movie based on the given combiner.
    Currently, only 'sum' combiner is considered, which sums up all types of a movie.
    """
    with tf.name_scope("movie_categories_layers"):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform([movie_categories_max, embed_dim], -1, 1), name="movie_categories_embed_matrix")
        movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories, name="movie_categories_embed_layer")
        if combiner == "sum":
            movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keep_dims=True)
    return movie_categories_embed_layer


def get_movie_cnn_layer(movie_titles, dropout_keep_prob, window_sizes=[3, 4, 5, 6]):
    """
    Implement convolutional neural network for movie titles.
    window_sizes: Text convolution sliding windows with sizes of 3, 4, 5, 6 words.
    """
    # Get word embedding vectors for each word in movie titles from the embedding matrix
    with tf.name_scope("movie_embedding"):
        movie_title_embed_matrix = tf.Variable(tf.random_uniform([movie_title_max, embed_dim], -1, 1), name="movie_title_embed_matrix")
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles, name="movie_title_embed_layer")
        # Add an extra dimension to movie_title_embed_layer
        # Here, the extra dimension is added at the last position, which represents the channel
        # So, the number of channels is 1
        # This is similar to how it is done for images
        movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

    # Apply convolution and max-pooling with different kernel sizes
    pool_layer_lst = []
    for window_size in window_sizes:
        with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
            # [window_size, embed_dim, 1, filter_num] represents the input channel is 1, and the output channel is filter_num
            filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num], stddev=0.1), name="filter_weights")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")

            # conv2d applies the convolutional kernel with size [filter_height * filter_width * in_channels, output_channels]
            # In this case, the convolutional kernel slides in two directions
            # conv1d slides the convolutional kernel in one direction only, which is the difference between conv1d and conv2d
            # strides require the first and last elements to be 1, and the default order for the four elements is NHWC (batch, height, width, channels)
            # padding is set to VALID, which means no padding; setting it to SAME would keep the input and output dimensions the same
            conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID", name="conv_layer")
            # tf.nn.bias_add adds the bias filter_bias to conv_layer
            # tf.nn.relu sets the activation function to ReLU
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")

            # tf.nn.max_pool takes the following arguments: input, pool_size (max-pooling window size), strides, and padding
            # This pooling operation converts the result of each convolutional kernel to a single element
            # Since there are 8 convolutional kernels here, the result is a vector with 8 elements
            maxpool_layer = tf.nn.max_pool(relu_layer, [1, sentences_size - window_size + 1, 1, 1], [1, 1, 1, 1], padding="VALID", name="maxpool_layer")
            pool_layer_lst.append(maxpool_layer)

    # Dropout layer
    with tf.name_scope("pool_dropout"):
        # The final result will be like this:
        # Suppose the window size of the convolutional kernel is 2, and there are 8 kernels
        # After the pooling operation, the result will be a vector with 8 elements
        # Each window size of the convolutional kernel, after pooling, will generate such a vector with 8 elements
        # So, the final result will be an 8-dimensional two-dimensional matrix, with the other dimension being the number of different window sizes
        # Here, it is 2, 3, 4, 5, so the final matrix will be 8*4
        pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
        max_num = len(window_sizes) * filter_num
        # Flatten the 8*4 matrix into a 32-element one-dimensional matrix
        pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")

        dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name="dropout_layer")
    return pool_layer_flat, dropout_layer


def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
    """
    Combine movie ID, movie genres, and movie title representations in separate small neural networks
    Then concatenate the outputs of each neural network to form the movie feature representation
    """
    with tf.name_scope("movie_fc"):
        # First fully connected layer
        movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, embed_dim, name="movie_id_fc_layer", activation=tf.nn.relu)
        movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, embed_dim, name="movie_categories_fc_layer", activation=tf.nn.relu)

        # Second fully connected layer
        movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  # (?, 1, 96)
        movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, 200, tf.tanh)  # (?, 1, 200)

        movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
    return movie_combine_layer, movie_combine_layer_flat
