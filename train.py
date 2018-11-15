import tensorflow as tf
import argparse
import json
import glob
import os
import math

from input_pipeline import get_input_fn
from model.model import get_model_fn
from model.common import get_optimizer
#from model.hooks import EvalSampleHook


def update_conf(config, jsons):
    '''
    config file의 내용에 fix_config의 내용을 update해나감
    '''
    for k, v in jsons.items():
        newKey = config[k]
        if type(v) is dict:
            update_conf(newKey, v)
        else:
            config[k] = v


def get_exp(args, config):
    batch_size = args.batch_size
    model_dir  = args.output

    train_conf = config['train']
    model_conf = config['model']

    hooks = []
    def experiment_fn(output_dir):
        run_config = tf.contrib.learn.RunConfig(model_dir=model_dir)
        optimizer_fn = get_optimizer(train_conf['start_lr']*math.sqrt(args.num_gpus),
                                     train_conf['every_step'],
                                     train_conf['coefficient'])

        model_fn = get_model_fn(optimizer_fn,
                                model_conf,
                                num_gpus=args.num_gpus)

        estimator = tf.estimator.Estimator(model_fn=model_fn,
                                           config=run_config)

        return tf.contrib.learn.Experiment(estimator=estimator,
                                           train_input_fn=get_input_fn(batch_size*args.num_gpus),
                                           eval_input_fn=get_input_fn(batch_size, isTrain=False),
                                           train_steps=train_conf['iteration'],
                                           eval_steps=train_conf['eval_steps'],
                                           train_monitors=hooks)
    return experiment_fn


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser(description='mnist trainer')
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='batch_size')
    parser.add_argument('--config',
                        default="config/default_config.json",
                        type=str,
                        help='config file')
    parser.add_argument('--num_gpus',
                        default=1,
                        type=int,
                        help='the number of gpus')
    parser.add_argument('--output',
                        default='trained_model',
                        type=str,
                        help='directory for trained_model')
    parser.add_argument('--fix_config',
                        default=None,
                        type=str,
                        help='')
    args = parser.parse_args()
    ####################################################################

    with open(args.config) as f:
        config = json.loads(f.read())['config']
    if args.fix_config is not None:
        jsons = json.loads(args.fix_config)
        update_conf(config, jsons)

    exp_fn = get_exp(args, config)

    tf.contrib.learn.learn_runner.run(experiment_fn=exp_fn,
                                      output_dir=args.output)
