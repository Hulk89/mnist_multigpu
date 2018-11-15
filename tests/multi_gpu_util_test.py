import sys
sys.path.append('')
from utils.multi_gpu_util import *
import tensorflow as tf

class MultiGpuUtilTest(tf.test.TestCase):
    def test_split_features(self):
        '''
        feature split을 잘 하는지..
        '''
        features = {'images': tf.constant([[4*i, 4*i+1, 4*i+2, 4*i+3] for i in range(8)]),
                    'labels': tf.constant([[4*i+3, 4*i+2, 4*i+1, 4*i] for i in range(8)]),
                    'tests':  tf.constant([[4*i, 3*i, 2*i, i] for i in range(8)])}

        with self.test_session():
            splited_features = split_features(features, 4)
            for i, splited_feature in enumerate(splited_features):
                #print(splited_feature['images'].eval())
                #print(splited_feature['labels'].eval())
                self.assertAllEqual(splited_feature['images'].eval(),
                                [[4*(2*i+j), 4*(2*i+j)+1, 4*(2*i+j)+2, 4*(2*i+j)+3]
                                            for j in range(2)])
                self.assertAllEqual(splited_feature['labels'].eval(),
                                [[4*(2*i+j)+3, 4*(2*i+j)+2, 4*(2*i+j)+1, 4*(2*i+j)]
                                            for j in range(2)])
                self.assertAllEqual(splited_feature['tests'].eval(),
                                [[4*(2*i+j), 3*(2*i+j), 2*(2*i+j), (2*i+j)]
                                            for j in range(2)])

    def test_merge_predict(self):
        #test_split_features와 의존도가 높다
        features = {'images': tf.constant([[4*i, 4*i+1, 4*i+2, 4*i+3] for i in range(8)]),
                    'labels': tf.constant([[4*i+3, 4*i+2, 4*i+1, 4*i] for i in range(8)]),
                    'tests':  tf.constant([[4*i, 3*i, 2*i, i] for i in range(8)])}

        with self.test_session():
            splited_features = split_features(features, 4)
            merged_predict = merge_predictions(splited_features)
            for key in features.keys():
                self.assertAllEqual(features[key].eval(),
                                    merged_predict[key].eval())

if __name__ == '__main__':
    tf.test.main()
