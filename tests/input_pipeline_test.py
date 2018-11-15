import sys
sys.path.append('')
from input_pipeline import get_input_fn
import tensorflow as tf

class InputTest(tf.test.TestCase):
    def test_innput_fn_basic(self):
        '''
        하나의 input을 잘 가져오는지 테스트
        '''
        images, labels = get_input_fn(1)()
        with self.test_session():
            imgs, lbls = [images['images'], labels['labels']]
            self.assertEqual(imgs.eval().shape, (1, 784))
            self.assertEqual(lbls.eval().shape, (1, 10))


    def test_innput_fn_different_batch(self):
        '''
        전체 data 갯수의 반보다 적게 batch를 꺼내면 data는 서로
        달라야 한다.
        '''
        images, labels = get_input_fn(100)()
        with self.test_session():
            imgs1, lbls1 = [images['images'], labels['labels']]
            imgs2, lbls2 = [images['images'], labels['labels']]
            self.assertNotAllClose(imgs1.eval(), imgs2.eval())


    def test_innput_fn_many(self):
        '''
        mnist의 data 갯수는 60000개이므로, 40000개 배치를 두번 보면 똑같은
        batch를 뱉어내야한다.
        '''
        images, labels = get_input_fn(40000)()
        with self.test_session():
            imgs1, lbls1 = [images['images'], labels['labels']]
            imgs2, lbls2 = [images['images'], labels['labels']]
            self.assertAllEqual(imgs1.eval(), imgs2.eval())



if __name__ == '__main__':
    tf.test.main()
