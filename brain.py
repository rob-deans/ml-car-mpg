import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Read back in the normalised data
df = pd.read_csv('norm_auto_mpg.csv', sep=',')

# Sort the data into output and input
input_data = [[row['cylinders'], row['displacement'], row['horsepower'], row['weight'], row['acceleration'],
               row['model year'], row['origin']] for _, row in df.iterrows()]

output_data = [[row['mpg']] for _, row in df.iterrows()]

TRAINING_SIZE = 0.8
INPUT_SIZE = len(input_data[0])
OUTPUT_SIZE = len(output_data[0])
HIDDEN_NEURONS = 14

np.random.seed(1)

# Sort the data into training and testing inputs and outputs
training_input = input_data[int(len(input_data) * TRAINING_SIZE):]
testing_input = input_data[:int(len(input_data) * TRAINING_SIZE)]

training_output = output_data[int(len(input_data) * TRAINING_SIZE):]
testing_output = output_data[:int(len(input_data) * TRAINING_SIZE)]

# Set up our placeholder's i.e. the inputs and the output
input_x = tf.placeholder(np.float32, [None, INPUT_SIZE], name='input_x')
output_x = tf.placeholder(np.float32, [None, OUTPUT_SIZE], name='output_x')

# Hidden layer stuff
hidden_W = tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_NEURONS]))
hidden_B = tf.Variable(tf.random_normal([HIDDEN_NEURONS]))

hidden_output = tf.sigmoid(tf.matmul(input_x, hidden_W) + hidden_B)

# Hidden layer 2
hidden_W2 = tf.Variable(tf.random_normal([HIDDEN_NEURONS, HIDDEN_NEURONS]))
hidden_B2 = tf.Variable(tf.random_normal([HIDDEN_NEURONS]))

hidden_output2 = tf.sigmoid(tf.matmul(hidden_output, hidden_W2) + hidden_B2)

# Output layer stuff
output_W = tf.Variable(tf.random_normal([HIDDEN_NEURONS, OUTPUT_SIZE]))
output = tf.sigmoid(tf.matmul(hidden_output2, output_W))

# Loss function:
cost = tf.reduce_mean(tf.square(output_x - output))
optimiser = tf.train.AdamOptimizer(0.01)
train = optimiser.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

loss_ = []
res = []

for i in range(10000):
    c_values = sess.run([train, cost, hidden_W, output_x], feed_dict={input_x: input_data, output_x: output_data})

    # Append the loss to an array so we can see how the loss goes down
    loss_.append(c_values[1])

for j, val in enumerate(testing_input):
    conf = sess.run(output, feed_dict={input_x: [val]}).tolist()

    # Find the difference to see how far off we are
    res.append(1/testing_output[j][0] - 1/conf[0][0])

print(np.mean(res))
print(np.max(res) - np.min(res))
# plt.plot([i for i in range(len(loss_))], loss_)
plt.plot([i for i in range(len(res))], res)
plt.show()
