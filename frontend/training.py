import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import pickle

import matplotlib.pyplot as plt
import time
import datetime

import movie_nn
import user_nn

tf.reset_default_graph()
train_graph = tf.Graph()

features = pickle.load(open('./data/features.p', 'rb'))
target_values = pickle.load(open('./data/targets.p', 'rb'))

# Hyperparameters
# Number of Epochs
num_epochs = 0
# Batch Size
batch_size = 256
dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 50
# Flag to determine whether to sum or average the movie category embedding vectors; considered using mean, but not implemented
combiner = "sum"
title_length = 15
save_dir = './save_model/'


def get_targets():
    targets = tf.placeholder(tf.int32, [None, 1], name="targets")
    # lr = tf.placeholder(tf.float32, name="LearningRate")
    return targets


# Custom method to get batches
def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]


with train_graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=True)
    targets = get_targets()

    # Get input placeholders
    uid, user_gender, user_age, user_job = user_nn.get_inputs()
    movie_id, movie_categories, movie_titles, dropout_keep_prob = movie_nn.get_inputs()
    # Get 4 embedding vectors for the User
    uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = user_nn.get_user_embedding(uid, user_gender,
                                                                                                       user_age,
                                                                                                       user_job)
    # Get user features
    user_combine_layer, user_combine_layer_flat = user_nn.get_user_feature_layer(uid_embed_layer, gender_embed_layer,
                                                                                 age_embed_layer, job_embed_layer)
    # Get embedding vector for movie IDs
    movie_id_embed_layer = movie_nn.get_movie_id_embed_layer(movie_id)
    # Get embedding vector for movie categories
    movie_categories_embed_layer = movie_nn.get_movie_categories_embed_layer(movie_categories, combiner)
    # Get feature vector for movie titles
    pool_layer_flat, dropout_layer = movie_nn.get_movie_cnn_layer(movie_titles, dropout_keep_prob)
    # Get movie features
    movie_combine_layer, movie_combine_layer_flat = movie_nn.get_movie_feature_layer(movie_id_embed_layer,
                                                                                     movie_categories_embed_layer,
                                                                                     dropout_layer)
    # Calculate the predicted rating
    # Note that there are two different schemes for inference, and their names (name values) are different. This is important for recommending movies later on.
    # TensorFlow's name_scope specifies the scope of the tensor, making it easy to access tensors later on by specifying the name_scope
    with tf.name_scope("inference"):
        # Directly multiply the user feature matrix with the movie feature matrix to get the score; the goal is to regress this score to the actual rating
        inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
        inference = tf.expand_dims(inference, axis=1)

    with tf.name_scope("loss"):
        # Calculate Mean Squared Error (MSE) loss to regress the predicted value to the actual rating
        cost = tf.losses.mean_squared_error(targets, inference)
        # Sum the cost over each dimension and compute the average to get the final loss
        loss = tf.reduce_mean(cost)
   
    train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

losses = {'train': [], 'test': []}

with tf.Session(graph=train_graph) as sess:
    # Collect data for TensorBoard
    

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print(f"Writing to {out_dir}\n")

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", loss)

    # Train Summaries
    # train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_op = tf.summary.merge([loss_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Inference summaries
    inference_summary_op = tf.summary.merge([loss_summary])
    inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
    inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch_i in range(num_epochs):

        # Split the dataset into training and testing sets with a random seed
        train_X, test_X, train_y, test_y = train_test_split(features, target_values, test_size=0.2, random_state=0)

        train_batches = get_batches(train_X, train_y, batch_size)
        test_batches = get_batches(test_X, test_y, batch_size)

        # Training iteration, save training loss
        for batch_i in range(len(train_X) // batch_size):
            x, y = next(train_batches)

            categories = np.zeros([batch_size, 18])
            for i in range(batch_size):
                categories[i] = x.take(6, 1)[i]

            titles = np.zeros([batch_size, title_length])
            for i in range(batch_size):
                titles[i] = x.take(5, 1)[i]

            feed = {
                uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                movie_categories: categories,  # x.take(6,1)
                movie_titles: titles,  # x.take(5,1)
                targets: np.reshape(y, [batch_size, 1]),
                dropout_keep_prob: dropout_keep
            }

            step, train_loss, summaries, _ = sess.run([global_step, loss, train_summary_op, train_op], feed)  # cost
            losses['train'].append(train_loss)
            train_summary_writer.add_summary(summaries, step)  #

            # Show every <show_every_n_batches> batches
            if batch_i % show_every_n_batches == 0:
                time_str = datetime.datetime.now().isoformat()
                print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(train_X) // batch_size),
                    train_loss))

        # Testing iteration using test data
        for batch_i in range(len(test_X) // batch_size):
            x, y = next(test_batches)

            categories = np.zeros([batch_size, 18])
            for i in range(batch_size):
                categories[i] = x.take(6, 1)[i]

            titles = np.zeros([batch_size, title_length])
            for i in range(batch_size):
                titles[i] = x.take(5, 1)[i]

            feed = {
                uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                movie_categories: categories,  # x.take(6,1)
                movie_titles: titles,  # x.take(5,1)
                targets: np.reshape(y, [batch_size, 1]),
                dropout_keep_prob: 1
            }

            step, test_loss, summaries = sess.run([global_step, loss, inference_summary_op], feed)  # cost

            # Save test loss
            losses['test'].append(test_loss)
            inference_summary_writer.add_summary(summaries, step)  #

            time_str = datetime.datetime.now().isoformat()
            if batch_i % show_every_n_batches == 0:
                print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(test_X) // batch_size),
                    test_loss))

    # Save Model
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
