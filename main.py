import numpy as np
import tensorflow as tf
from tensorflow import feature_column
import pandas as pd
from packdatefeatures.packdate import PackDateFeatures

train_file_path = "/home/denis/train_waiters.csv"
test_file_path = "/home/denis/eval_waiters.csv"

np.set_printoptions(precision=3, suppress=True)

LABEL_COLUMN = 'is_waiter'
LABELS = [True, False]


def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=50,
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs)
    return dataset


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


SELECT_COLUMNS = ['is_waiter', 'uuid', 'position', 'organization', 'description', 'start', 'end']
temp_dataset = get_dataset(train_file_path,
                           select_columns=SELECT_COLUMNS)


example_batch, labels_batch = next(iter(temp_dataset))


def pack(features, label):
    return tf.stack(list(features.values()), axis=-1), label


packed_dataset = temp_dataset.map(pack)

for features, labels in packed_dataset.take(1):
    print(features.numpy())
    print()
    print(labels.numpy())


example_batch, labels_batch = next(iter(temp_dataset))

DATE_FEATURES = ['start', 'end']

packed_train_data = raw_train_data.map(
    PackDateFeatures(DATE_FEATURES))

packed_test_data = raw_test_data.map(
    PackDateFeatures(DATE_FEATURES))


packed_dataset = temp_dataset.map(pack)

emb = feature_column.categorical_column_with_vocabulary_list(
      'emb', ['официан*', 'бар*', 'ресторан*'])


emp_c = feature_column.embedding_column(emb, dimension=8)

preprocessing_layer = tf.keras.layers.DenseFeatures(emp_c)

print(preprocessing_layer(example_batch).numpy()[0])

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model.fit(train_data, epochs=50)

test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

predictions = model.predict(test_data)

for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    print("Is waiter: {:.2%}".format(prediction[0]),
          " | Actual outcome: ",
          ("Waiter" if bool(survived) else "Not waiter"))

