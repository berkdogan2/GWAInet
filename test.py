import tensorflow as tf
from model import GWAInet
from argparse import Namespace
import numpy as np

args=Namespace()

checkpoint='/GWAInet_test/checkpoint/model.ckpt-54000'
args.npy_test_HR_path='/GWAInet_test/y_sample.npy'
args.npy_test_LR_path='/GWAInet_test/x_sample.npy'
args.npy_test_GHR_path='/GWAInet_test/xg_sample.npy'
args.result_dir='/results'

args.training_LR_RGB_mean=np.array([130.98089974, 106.99848177, 94.87540804])
args.training_HR_RGB_mean=np.array([130.99337721, 107.01429743, 94.89514401])

args.img_height=32 
args.img_width=32 
args.scale=8
args.output_channels=3
args.scaling_factor=1
args.batch_size = 8

args.num_layers=16
args.feature_size=64
args.merge_resblock=4

net = GWAInet(args)

var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator') + \
           tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='warper')                      
weight_initializer = tf.train.Saver(var_list)

net.sess.run(tf.global_variables_initializer())

print('Loading weights from the pre-trained model')
try:
    weight_initializer.restore(net.sess, checkpoint)
    print("Restore weights success!")
except:
    print("Restore failure!")

net.set_test_data()
net.print_test()