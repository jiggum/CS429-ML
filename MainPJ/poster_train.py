#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Model training for Iris data set using Validation Monitor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
IRIS_TRAINING = "poster_data/poster_training.csv"
IRIS_TEST = "poster_data/poster_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float)


# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=12)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[20, 60, 30],
                                            n_classes=3,
                                            model_dir="poster_data/save/20_60_30",
                                            config=tf.contrib.learn.RunConfig(
                                                save_checkpoints_secs=5))

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=0)
               #monitors=[validation_monitor])

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]

print("Accuracy: {0:f}".format(accuracy_score))

# Classify two new flower samples.

new_samples = np.array(
    [
        [0.000042,0,0,0,0.048689,0.000040,3,23,8,17,0.000063,0.062272],#0
        [0.000185,0,0,0,0.078222,0.000203,80,98,4,23,0.000895,0.477333],#0
        [0.000332,0,0,0,0.241667,0.000132,147,166,0,44,0.143626,0.921637],#0
        [0.000139,0,0,0,0.051163,0.000092,19,22,3,10,0.001085,0.225032],#0
        [0.000574,0,0,0,0.196154,0.000504,0,34,0,0,0.014028,0.288328],#0
        [0.000072,1,0,0,0.078571,0.000079,35,51,0,9,0.000198,0.196190],#0
        [0.000240,0,0,0,0.100858,0.000208,0,16,0,8,0.002989,0.147222],#0
        [0.000015,0,0,0,0.051995,0.000014,149,196,0,28,0.000010,0.219498],#0
        [0.000571,0,0,0,0.084323,0.000694,16,20,0,3,0.008545,0.598993],#0
        [0.000465,0,0,0,0.052456,0.000393,5,5,0,0,0.005772,0.115124],#0
        #############
        [0.001197,0,0,0,0.235096,0.000704,30,62,0,11,0.356235,0.863962],#1
        [0.000425,0,0,0,0.064946,0.000353,12,22,0,2,0.009930,0.393905],#1
        [0.000376,0,0,0,0.179779,0.000038,33,256,7,78,0.267276,0.953838],#1
        [0.000993,0,0,0,0.093692,0.001057,9,13,2,1,0.016600,0.331165],#1
        [0.002571,0,0,0,0.084926,0.003027,5,7,0,0,0.231276,0.327690],#1
        [0.000571,0,0,0,0.077791,0.000654,23,27,0,37,0.024757,0.664987],#1
        [0.000318,0,0,0,0.048889,0.000342,37,43,1,6,0.002202,0.739259],#1
        [0.001215,0,0,0,0.060986,0.001503,11,17,0,2,0.068376,0.653595],#1
        [0.000684,0,0,0,0.450242,0.000122,0,174,0,0,0.494720,1.019769],#1
        [0.000438,0,0,0,0.160159,0.000189,81,146,0,17,0.444796,0.895111],#1
        #############
        [0.000306,1,1,0,0.052121,0.000288,15,26,2,3,0.003693,0.472941],#2
        [0.000780,0,0,0,0.287628,0.000242,0,107,0,0,0.142254,0.933468],#2
        [0.000577,0,0,0,0.048333,0.000375,6,10,0,4,0.034880,0.438914],#2
        [0.000124,0,0,0,0.071823,0.000038,34,72,0,13,0.001842,0.295833],#2
        [0.000141,0,0,1,0.058031,0.000107,41,53,10,24,0.000914,0.377143],#2
        [0.000562,0,0,0,0.078125,0.000090,9,132,2,16,0.255040,0.887611],#2
        [0.000203,0,0,1,0.127496,0.000158,33,103,24,29,0.004326,0.752013],#2
        [0.000252,1,1,1,0.117488,0.000222,37,83,12,22,0.003449,0.696500],#2
        [0.000342,0,0,0,0.097778,0.000341,0,28,0,0,0.003612,0.273333],#2
        [0.000106,1,1,1,0.068493,0.000066,79,139,28,47,0.005384,0.572500],#2


    ], dtype=float)
y_p = list(classifier.predict_proba(new_samples, as_iterable=True))
y = list(classifier.predict(new_samples, as_iterable=True))
for i, elem in enumerate(y):
    print(elem, end="")
    if i%10==9:
        print("")
for i, elem in enumerate(y_p):
    print(elem, end="")
    if i%10==9:
        print("")
