from datetime import datetime

import tensorflow as tf


class PackDateFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        date_features = [features.pop(name) for name in self.names]
        date_features = [tf.cast(feat, datetime.date) for feat in date_features]
        date_features = tf.stack(date_features, axis=-1)
        features['date'] = date_features

        return features, labels
