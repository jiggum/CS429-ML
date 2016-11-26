import cv2
from parse_poster import *
import numpy as np
import tensorflow as tf
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=12)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[20, 60, 30],
                                            n_classes=3,
                                            model_dir="save/20_60_30",
                                            config=tf.contrib.learn.RunConfig(
                                                save_checkpoints_secs=5))

for i in range(0,34):
    img_origin = cv2.imread("image/"+ str(i) + ".jpg", 0)
    imgs, rects, resize_ratio, padding = detect_letter_boxs(img_origin)
    #show_img(imgs[-1])
    figures = get_figures_of_blocks(img_origin, rects, resize_ratio, padding)
    #save_figures_to_text(figures, str(i))
    samples = figures_array(figures)
    new_samples = np.array(samples, dtype=float)
    y_p = list(classifier.predict_proba(new_samples, as_iterable=True))
    for i, elem in enumerate(y_p):
        print elem
    img_result = detect_boxs_type(img_origin, rects, y_p)
    img_result_maximum = detect_boxs_type_maximum(img_origin, rects, y_p)
    show_imgs([imgs[-1],img_result, img_result_maximum])