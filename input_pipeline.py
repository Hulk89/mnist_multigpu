import tensorflow as tf

def get_input_fn(batch_size, isTrain=True):
    def _mnist_generator(isTrain=True):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if isTrain:
            x, y = x_train, y_train
        else:
            x, y = x_test, y_test

        i = 0
        num_data = x.shape[0]

        while True:
            if (i+1)*batch_size >= num_data:
                i = 0
            start, end = i*batch_size, (i+1)*batch_size
            batch_xs, batch_ys = x[start:end], y[start:end]
            yield batch_xs, batch_ys


            i += 1
    def _input_fn():
        ds = tf.data.Dataset.from_generator(_mnist_generator,
                                            (tf.float32, tf.int32))

        value = ds.make_one_shot_iterator().get_next()
        images, labels = value

        normalized_images = images / 255.0
        reshaped_images = tf.reshape(normalized_images, [-1, 784])
        onehot_labels = tf.one_hot(labels, 10)

        return {"images": reshaped_images}, {"labels": onehot_labels}
    return _input_fn


