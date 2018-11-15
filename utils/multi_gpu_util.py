import tensorflow as tf

def split_features(features, num_gpus):
    '''
    features는 dictionary 형태로 들어온다.
    이 때, gpu 갯수로 dictionary를 쪼개주는 것이 이 곳에서 할 일!
    '''
    splited_features = [(key, tf.split(val, num_gpus, axis=0))
                            for key, val in features.items()]
    dicts = [{} for _ in range(num_gpus)]
    for splited_feature in splited_features:
        key, vals = splited_feature
        for i, val in enumerate(vals):
            dicts[i][key] = val
    return dicts

def merge_predictions(pred_dicts):
    '''
    각 gpu에서 prediction dictionary를 계산하고, 이를 모아서 pred_dicts로 넘어온다.
    여기서는 다시 하나로 합치는 일을 한다.
    '''
    merge_preds = {}
    for key in pred_dicts[0].keys():
        merge_preds[key] = []

    for pred_dict in pred_dicts:
        for key, value in pred_dict.items():
            merge_preds[key].append(value)
    for key in merge_preds.keys():
        merge_preds[key] = tf.concat(merge_preds[key], 0)

    return merge_preds

def average_gradients(tower_grads):
    '''
    [[(grad, var)...], ...]가 들어오면, [(grad, var)...] 로 바꿔줌
    '''
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        grad = tf.clip_by_norm(grad, 5.0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

