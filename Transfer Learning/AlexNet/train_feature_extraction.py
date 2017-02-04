import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle

nb_classes = 43
epochs = 5
batch_size = 64

# TODO: Load traffic signs data.
training_file = "train.p"
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']

# TODO: Split data into training and validation sets.
X_train, X_val, y_train, y_val = \
    train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, [None, 32, 32, 3])
labels = tf.placeholder(tf.float32, [None])
resized = tf.image.resize_images(features, (227, 227))
keep_prob = tf.placeholder(tf.float32)
beta = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
# Training Pipeline
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
loss_op  = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op, var_list=[fc8W, fc8b])

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

init_op = tf.global_variables_initializer()

def eval_on_data(X, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={features: X_batch, labels: y_batch})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])

    return total_loss/X.shape[0], total_acc/X.shape[0]

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            sess.run(train_op, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})

        val_loss, val_acc = eval_on_data(X_val, y_val, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")

# Epoch 1
# Time: 1996.921 seconds
# Validation Loss = 0.521746287808
# Validation Accuracy = 0.860499265809
#
# Epoch 2
# Time: 1945.048 seconds
# Validation Loss = 0.351755901719
# Validation Accuracy = 0.905402272196
#
# Epoch 3
# Time: 1951.288 seconds
# Validation Loss = 0.259610752313
# Validation Accuracy = 0.93708941958
#
# Epoch 4
# Time: 1953.500 seconds
# Validation Loss = 0.233335638783
# Validation Accuracy = 0.939021562713
#
# Epoch 5
# Time: 1952.720 seconds
# Validation Loss = 0.192396457643
# Validation Accuracy = 0.949841564263
#
# Epoch 6
# Time: 1952.301 seconds
# Validation Loss = 0.174743473909
# Validation Accuracy = 0.953319421898
#
# Epoch 7
# Time: 1951.954 seconds
# Validation Loss = 0.154698730207
# Validation Accuracy = 0.959193137023
#
# Epoch 8
# Time: 1951.899 seconds
# Validation Loss = 0.147947926187
# Validation Accuracy = 0.960043280029
#
# Epoch 9
# Time: 1953.008 seconds
# Validation Loss = 0.139560060333
# Validation Accuracy = 0.963366566218
#
# Epoch 10
# Time: 1949.261 seconds
# Validation Loss = 0.122237600339
# Validation Accuracy = 0.968158281165