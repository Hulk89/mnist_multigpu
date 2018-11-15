import tensorflow as tf
import sys
sys.path.append('')
from utils.multi_gpu_util import split_features, merge_predictions, average_gradients

def mnist_model(features, model_conf, reuse=False):
    dim = model_conf['dim']
    with tf.variable_scope("mnist_model", reuse=reuse) as scope:
        W1 = tf.get_variable("W1", [784, dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable("b1", [dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
        W2 = tf.get_variable("W2", [dim, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable("b2", [10], initializer=tf.truncated_normal_initializer(stddev=0.1))

        output = tf.matmul(tf.nn.relu(tf.matmul(features['images'], W1) + b1), W2) + b2
        return output

def get_predict(output):
    with tf.name_scope('prediction'):
        prediction = tf.nn.softmax(output)
        return {"prediction": prediction}


def get_model_fn(optimizer_fn,
                 model_conf,
                 num_gpus=1):
    def model_fn(features, labels, mode):
        def makeModel(features):
            '''
            features를 받아서 logits를 내보내는 부분.
            '''
            return mnist_model(features, model_conf, reuse=tf.AUTO_REUSE)

        if mode == tf.estimator.ModeKeys.TRAIN:  # TRAIN일 때 
            optimizer = optimizer_fn()

            feature_splits = split_features(features, num_gpus)
            label_splits = split_features(labels, num_gpus)

            losses = []
            tower_grads =[]
            predicts = []

            for i in range(num_gpus):
                with tf.device('/device:GPU:{}'.format(i)):
                    with tf.name_scope("model_{}".format(i)) as scope:
                        output = makeModel(feature_splits[i])  # i번째 feature들을 가지고 i번째 model을 만듬

                        pred_dict = get_predict(output)
                        predicts.append(pred_dict)

                        loss = tf.losses.softmax_cross_entropy(label_splits[i]['labels'],
                                                               output)
                        gradients = optimizer.compute_gradients(loss)  # [(gradient, var), ...]
                        tower_grads.append(gradients)
                        losses.append(loss)

            pred_dict = merge_predictions(predicts)

            global_step = tf.train.get_global_step()
            grads = average_gradients(tower_grads)
            apply_gradient_op = optimizer.apply_gradients(grads,
                                                          global_step=global_step)

            loss = tf.reduce_mean(losses)
            tf.summary.scalar("loss", loss)

            return tf.estimator.EstimatorSpec(mode,
                                              predictions=pred_dict,
                                              loss=loss,
                                              train_op=apply_gradient_op,
                                              eval_metric_ops=None)

        else:
            output = makeModel(features)
            pred_dict = get_predict(output)

            eval_metric_ops = None
            loss = None

            if mode == tf.estimator.ModeKeys.PREDICT:
                pass
            else: # tf.estimator.ModeKeys.EVAL
                loss = tf.losses.softmax_cross_entropy(labels['labels'],
                                                       output)
                eval_metric_ops = None  # 실제로는 여기에 eval metric을 넣을 수 있다.

            #TODO: evaluation_hooks, prediction_hooks
            return tf.estimator.EstimatorSpec(mode,
                                              predictions=pred_dict,
                                              loss=loss,
                                              train_op=None,
                                              eval_metric_ops=eval_metric_ops)
    return model_fn

if __name__ == '__main__':
    #TODO: test로 가야함.
    f = {'images': tf.Variable(tf.zeros([64, 784]))}
    o = mnist_model(f, None)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run(o)
        print(o.shape)
