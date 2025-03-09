import tensorflow as tf
import numpy as np
import os
N_NEIGHBORS=200

def parse_proto(example_proto):
    features = {
    'patches': tf.io.FixedLenFeature((N_NEIGHBORS*5,), tf.float32, default_value = tf.zeros([N_NEIGHBORS*5])),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    parsed_features['patches'] = tf.reshape(parsed_features['patches'],[N_NEIGHBORS, 5])
    return  parsed_features['patches']

def dataset(filenames):
    buffer_size = 1500*len(filenames)
    i=0
    for filename in filenames:
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(parse_proto)
        data_list = []
        for record in dataset:
            # Convert each feature dictionary to NumPy
            data_list.append(record)
        # Convert the list of dictionaries to a structured NumPy array or save directly
        np.save('numpy_training_data/famousthingi_logmap_patches_{}.npy'.format(i), data_list)
        i+=1

        

if __name__ == '__main__':
    if not os.path.exists('numpy_training_data'):
        os.mkdir('numpy_training_data')
    filenames = ["training_data/famousthingi_logmap_patches_{}.tfrecords".format(i) for i in range(0,56)]
    dataset = dataset(filenames)


