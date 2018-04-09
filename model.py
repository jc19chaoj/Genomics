import os
import sys

import numpy as np
import tensorflow as tf
import pandas as pd


LOGDIR = '/tmp/temp_log/'

def str_to_int(gene_str):
    gene_list = list(gene_str)
    gene_dict = {"A": 0, "C":0, "G":0, "T":0}
    index = 0
    for s in gene_list[::-1]:
        gene_dict[s] += 2**index
        index += 1

    gene_array = list(gene_dict.values())
    return np.asarray(gene_array)

def load_data(filename):
    input_file = filename

    df = pd.read_csv(input_file, header=0)
    df_array = df.as_matrix()
    df_array = np.delete(df_array,0,1)

    return df_array

def shuffle_data(raw_data):
    np.random.shuffle(raw_data)

    inputs = []
    labels = raw_data[:,1]
    labels = np.vstack((1-labels,labels))

    gene_seq = raw_data[:,0]
    for seq in gene_seq:
        inputs.append(str_to_int(seq))

    return np.asarray(inputs), labels.T

def next_batch(inputs, labels, batch_index, batch_size):
    """load next minibatch of inputs and labels"""
    inputs_size = len(inputs)
    #print("input size:", inputs_size)
    input_batch = inputs[batch_index%inputs_size : (batch_index%inputs_size+batch_size)]
    input_batch = np.asarray(input_batch).astype(np.float32)

    label_batch = labels[batch_index%inputs_size : (batch_index%inputs_size+batch_size)]
    label_batch = np.asarray(label_batch).astype(np.float32)
    #print(label_batch[0])
    return input_batch, label_batch


def fc_layer(name, inputs, output_units, activation=tf.nn.relu, dropout_rate=0.5):
    input_units = inputs.get_shape()[-1]

    with tf.name_scope(name):
        w_init = tf.constant_initializer(np.random.rand(input_units, output_units).astype(np.float32))
        b_init = tf.constant_initializer(np.zeros((1, output_units)).astype(np.float32))
        fc = tf.layers.dense(inputs, output_units, activation=activation,
                            name=name,
                            kernel_initializer=w_init,
                            bias_initializer=b_init)
        fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
        print(fc_vars)
        tf.summary.histogram('weights', fc_vars[0])
        tf.summary.histogram('biases', fc_vars[1])
        tf.summary.histogram('activation', fc)

        return tf.layers.dropout(fc,dropout_rate)

def nn_model(learning_rate, keep=1):
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data
    X = tf.placeholder(tf.float32, shape=[None, 4], name="x")
    print("Inputs:",X)
    #tf.summary.image('input', X, 3)
    y = tf.placeholder(tf.float32, shape=[None, 2], name="labels")

    # fc1
    fc1 = fc_layer('fc1', X, 48, dropout_rate=keep)
    tf.summary.histogram('fc1', fc1)
    # fc2
    # fc2 = fc_layer('fc2', fc1, 48)
    # tf.summary.histogram('fc2', fc2)
    # fc3
    #fc3 = fc_layer('fc3', fc2, 100)
    #tf.summary.histogram('fc3', fc3)
    # output_layer
    output_layer = fc_layer('output_layer', fc1, 2, activation=None)

    with tf.name_scope('cross_entropy'):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=output_layer, labels=y), name='xent')
        tf.summary.scalar('xent', xent)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate,0.8).minimize(xent)

    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(output_layer,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #accuracy = tf.reduce_mean(correct_pred)
        tf.summary.scalar('accuracy', accuracy)

    summary_merge = tf.summary.merge_all()

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR+"lr_%.0E"%learning_rate)
    writer.add_graph(sess.graph)

    raw_data = load_data("train.csv")
    inputs, labels = shuffle_data(raw_data)
    batch_size = 100
    batch_index = 0

    for i in range(200000):

        #print(inputs,labels)
        inputs_batch, labels_batch = next_batch(inputs, labels, batch_size, batch_index)
        batch_index += batch_size
        # check training accuracy every 200 iterations
        if i % 40 == 0:
            [loss, s] = sess.run(
                [xent, summary_merge],
                feed_dict={X: inputs_batch, y: labels_batch})
            writer.add_summary(s, i)
        if i % 200 == 0:
            inputs, labels = shuffle_data(raw_data)
            train_accuracy = sess.run(
                accuracy,
                feed_dict={X: inputs, y: labels})
            print("Step:", i, "accuracy =", train_accuracy, "loss =", loss)

        sess.run(optimizer, feed_dict={X: inputs_batch, y: labels_batch})


    t_raw_input = load_data("test.csv")
    t_inputs = []
    #print(t_raw_input)
    for i in t_raw_input:
        #print(i)
        t_inputs.append(str_to_int(i[0]))

    t_inputs = np.asarray(t_inputs)
    test_result = sess.run(output_layer, feed_dict={X: t_inputs})
    test_result = np.argmax(test_result,1).reshape(-1,1)
    #print(test_result, test_result.shape)
    df = pd.DataFrame(test_result)
    #df = pd.DataFrame([df], columns = ["id", "prediction"])
    df.to_csv("test_result.csv")

def main():
    for learning_rate in [1E-6]:

        print('Starting run for %s' % learning_rate)

        # run model
        nn_model(learning_rate, keep=0.5)

    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

if __name__ == '__main__':
    main()
