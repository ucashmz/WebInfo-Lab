import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

pred = [[0.1, 0.2]]
label = [[0, 0]]

pred1 = [[0.1]]
label1 = [[0]]
pred2 = [[0.2]]
label2 = [[0]]

print(tf.losses.binary_crossentropy(label, pred))
print((tf.losses.binary_crossentropy(label1, pred1) +
      tf.losses.binary_crossentropy(label2, pred2)) / 2)
