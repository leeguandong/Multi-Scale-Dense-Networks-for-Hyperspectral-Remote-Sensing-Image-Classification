# 3D-CNN模型
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.layers.convolutional import conv3d, conv2d
from tensorflow.python.layers.pooling import max_pooling3d, max_pooling2d
from tensorflow.python.layers.core import dense, dropout
from tensorflow.contrib.layers import flatten, l2_regularizer
from tensorflow.python.ops.nn import relu, elu
import Utils

# Define global variables

NUM_CLASSES = Utils.classes
IMAGE_SIZE = Utils.patch_size

KERNEL_SIZE1 = Utils.kernel_conv1  # before it was 5 for 37x37
KERNEL_SIZE2 = Utils.kernel_conv2
STRIDE_CONV1 = Utils.stride_conv1
STRIDE_CONV2 = Utils.stride_conv2
POOL_SIZE1 = Utils.pool_size1
POOL_SIZE2 = Utils.pool_size2
DROP_RATE = Utils.drop
CONV1_FILTERS = Utils.filters_conv1
CONV2_FILTERS = Utils.filters_conv2
FC1 = Utils.fc1
FC2 = Utils.fc2
LR = Utils.learning_rate
RReLU_min = 0.1
RReLU_max = 0.3
REG_lambda = 0.01


# Build the model up to where it may be used for inference.
def cnn_3d(images, is_training):
    """
    Build the model for 2D-CNN.

    Inputs:
    -- images: Images placeholder
    -- is_training: bool placeholder, training or not

    Output:
    -- Logits: Return the output of the model

    """
    # Size of images & labels
    height = int(images.shape[1])
    width = int(images.shape[2])
    depth = int(images.shape[3])

    images = tf.reshape(images, [-1, height, width, depth, 1])

    # Build the model
    with tf.name_scope('CONV1'):
        l_conv1 = conv3d(images, filters=4, kernel_size=[3, 3, 10], strides=[1, 1, 5],
                                                activation=relu,
                         kernel_regularizer=l2_regularizer(REG_lambda))
        # l_conv1 = rrelu(l_conv1, is_training)
        l_maxpool1 = max_pooling3d(l_conv1, pool_size=[3, 3, 3], strides=[1, 1, 1], padding='same', name='Maxpool1')

    with tf.name_scope('CONV2'):
        l_conv2 = conv3d(l_maxpool1, filters=16, kernel_size=[3, 3, 10], strides=[1, 1, 2],
                                                  # activation=relu,
                         kernel_regularizer=l2_regularizer(REG_lambda))
        l_conv2 = rrelu(l_conv2, is_training)
        l_maxpool2 = max_pooling3d(l_conv2, pool_size=[3, 3, 3], strides=[1, 1, 1], padding='same', name='Maxpool2')

    l_flatten = flatten(l_maxpool2, scope='Flatten')

    with tf.name_scope('FC1'):
        l_fc1 = dense(l_flatten, 200, kernel_regularizer=l2_regularizer(REG_lambda),
                                            # activation=relu
                      )
        l_fc1 = rrelu(l_fc1, is_training)
        l_drop1 = dropout(l_fc1, Utils.drop, training=is_training, name='Dropout1')

    with tf.name_scope('FC2'):
        l_fc2 = dense(l_drop1, 200, kernel_regularizer=l2_regularizer(REG_lambda),
                                            # activation=relu
                      )
        l_fc2 = rrelu(l_fc2, is_training)
        l_drop2 = dropout(l_fc2, Utils.drop, training=is_training, name='Dropout2')

    logits = dense(l_drop2, NUM_CLASSES, name='Output')

    return logits


# 2D-CNN model
def cnn_2d(images, is_training):
    """
    Build the model for 2D-CNN.

    Inputs:
    -- images: Images placeholder
    -- is_training: bool placeholder, training or not

    Output:
    -- Logits: Return the output of the model

    """
    # Build the CNN model
    l_conv1 = conv2d(images, CONV1_FILTERS, KERNEL_SIZE1,
                     strides=STRIDE_CONV1, activation=relu, name='Conv1')

    l_maxpool1 = max_pooling2d(l_conv1, POOL_SIZE1, POOL_SIZE1,
                               padding='same', name='Maxpool1')

    l_conv2 = conv2d(l_maxpool1, CONV2_FILTERS, KERNEL_SIZE2,
                     strides=STRIDE_CONV2, activation=relu, name='Conv2')

    l_maxpool2 = max_pooling2d(l_conv2, POOL_SIZE2, POOL_SIZE2,
                               padding='same', name='Maxpool2')

    l_flatten = flatten(l_maxpool2, scope='Flatten')

    l_fc1 = dense(l_flatten, FC1, activation=relu, name='Fc1')

    l_drop = dropout(l_fc1, DROP_RATE, training=is_training,
                     name='Dropout')

    l_fc2 = dense(l_drop, FC2, activation=relu, name='Fc2')

    logits = dense(l_fc2, NUM_CLASSES, name='Output')

    return logits


def cnn_test(images, is_training):
    # Build the CNN model
    l_conv1 = conv2d(images, 10, [7, 7],
                     strides=[1, 1],
                     # activation=rrelu,
                     name='Conv1')

    l_conv1 = rrelu(l_conv1, is_training)
    l_flatten = flatten(l_conv1, scope='Flatten')

    logits = dense(l_flatten, NUM_CLASSES, name='Output')

    return logits


# Build the model up to where it may be used for inference.
def inference(model, images, is_training):
    if model == 'cnn_2d':
        logits = cnn_2d(images, is_training)
        return logits
    elif model == 'cnn_3d':
        logits = cnn_3d(images, is_training)
        return logits
    elif model == 'cnn_test':
        logits = cnn_test(images, is_training)
        return logits
    else:
        print('Cannot find the model, plz check your spell.')


# Define the loss function
def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
    Returns:
    loss: Loss tensor of type float.
    """
    #    labels = tf.to_int64(labels, name='')
    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='cross-entropy-mean')

        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('cross-entropy-loss', loss)
    return loss


# Define the Training OP
def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    Returns:
    train_op: The Op for training.
    """
    with tf.name_scope('Training'):
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdagradOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# Define the Evaluation OP
def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
    Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    with tf.name_scope('Evaluation'):
        correct = tf.nn.in_top_k(logits, labels, 1, name='correct-prediction')
        # Return the number of true entries.
        with tf.name_scope('accuracy'):
            result = tf.cast(correct, tf.int32)
            result = tf.reduce_sum(result)
        tf.summary.scalar('mini-batch-accuracy', result)
    return result

def pred(logits):
     with tf.name_scope("Prediction"):
         prediction=tf.nn.softmax(logits,name="test-prediction")
     return prediction

# Activation functions
def rrelu(features, is_training, lower_bound=0.125, upper_bound=0.334, scope='rrelu'):
    with tf.variable_scope(scope):
        def training_phase():
            alpha = tf.random_uniform([], minval=lower_bound, maxval=upper_bound, name='random')
            leaked = tf.scalar_mul(alpha, features)
            tensor = tf.maximum(leaked, features, name='maximum')
            return tensor

        def inference_phase():
            alpha = (lower_bound + upper_bound) / 2.0
            leaked = tf.scalar_mul(alpha, features)
            tensor = tf.maximum(leaked, features, name='maximum')
            return tensor

        return tf.cond(is_training, training_phase, inference_phase)


def lrelu(features, leaky_ratio, name='LReLU'):
    """Computes leaky rectified linear: max(features, FLAGS.leak_ratio * features)
    Args:
        - features: A Tensor.
        - name: A name for the operation (optional)
    Returns:
        A Tensor.
    """
    with tf.name_scope(name):
        leaked = tf.scalar_mul(leaky_ratio, features)
        tensor = tf.maximum(features, leaked, name='maximum')
    return tensor
