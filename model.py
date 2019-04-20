import os
import tensorflow as tf
import pickle
import numpy as np
import utils
import scipy.misc

class GWAInet(object):

    def __init__(self,args): 
        tf.reset_default_graph()
        print("Building the model...")   
        self.args=args
        
        with tf.variable_scope('placeholders'):
            self.is_training = tf.placeholder(tf.bool, [])
            self.input = tf.placeholder(tf.uint8,[None,self.args.img_height,self.args.img_width,self.args.output_channels])
            self.input_g = tf.placeholder(tf.uint8,[None,self.args.img_height*self.args.scale,self.args.img_width*self.args.scale,self.args.output_channels]) 
            self.input_upsampled = tf.placeholder(tf.uint8,[None,self.args.img_height*self.args.scale,self.args.img_width*self.args.scale,self.args.output_channels]) 
        
        with tf.variable_scope('preprocessing'):
            self.input_p = tf.cast(self.input, tf.float32)
            self.input_upsampled_p = tf.cast(self.input_upsampled, tf.float32)
            self.input_g_p = tf.cast(self.input_g, tf.float32)
            
            self.input_p_0_1 = self.input_p / 255    
            self.args.training_LR_RGB_mean = self.args.training_LR_RGB_mean / 255     
            self.input_g_p_0_1 = self.input_g_p / 255
            self.args.training_HR_RGB_mean = self.args.training_HR_RGB_mean / 255
            self.input_upsampled_p_0_1 = self.input_upsampled_p / 255
            
            self.input_p=self.input_p_0_1-self.args.training_LR_RGB_mean
            self.input_g_p=self.input_g_p_0_1-self.args.training_HR_RGB_mean   
            self.input_upsampled_p = self.input_upsampled_p_0_1-self.args.training_LR_RGB_mean
                
        self.flow_field = self.warper_11(self.input_upsampled_p, self.input_g_p, self.is_training)
        
        self.flow_field_t = tf.transpose(self.flow_field, (0,3,1,2))
        
        self.input_g_p_warped = utils.tf_warp3(self.input_g_p, self.flow_field_t, 256, 256)
 
        self.g_output = self.model_guidance_9(self.input_p, self.input_g_p_warped, False)

        with tf.variable_scope('deprocessing'):
            self.g_output_dp_0_1=(self.g_output + 1) / 2            
            self.g_output_dp=tf.image.convert_image_dtype(self.g_output_dp_0_1, dtype=tf.uint8, saturate=True)
     
        self.up_test=0
        self.down_test=0
        
        self.sess = tf.Session()
       
        if not os.path.exists(self.args.result_dir):
            os.makedirs(self.args.result_dir)
     
        if not os.path.exists(self.args.result_dir+'/sr'):
            os.makedirs(self.args.result_dir+'/sr')
        print("Done building!")
	
    def set_test_data(self):
        self.x_test=list(np.load(self.args.npy_test_LR_path))
        self.y_test=list(np.load(self.args.npy_test_HR_path))
        self.xg_test=list(np.load(self.args.npy_test_GHR_path))
        self.num_test_image=len(self.y_test)

    def get_batch_test(self, send_summary_batch=False, summary_size=5):
        if send_summary_batch:
            window=np.random.randint(0,self.num_test_image,summary_size)
        else:
            self.up_test=min(self.down_test+self.args.batch_size,len(self.y_test))
            window = [x for x in range(self.down_test,self.up_test)]
            self.down_test=self.up_test%len(self.y_test)
        x = [self.x_test[q] for q in window]
        y = [self.y_test[q] for q in window]
        x_g=[]
        x_upsampled = [scipy.misc.imresize(x_,(self.args.img_height*self.args.scale,self.args.img_width*self.args.scale),interp="bicubic") for x_ in x]
        
        x_g = [self.xg_test[q] for q in window]
        return (x,y,x_g,x_upsampled,window)
    
    def predict(self,x,x_g,x_u):
        return self.sess.run(self.g_output_dp,feed_dict={self.input:x, self.input_g:x_g, self.is_training:False, self.input_upsampled:x_u})
     
    def print_test(self):        
        num_test=self.num_test_image
    
        i = 0
        count=0
        while i<num_test:
            j = min(i + self.args.batch_size, num_test)
                
            x,y,x_g,x_u,window=self.get_batch_test()
            output=self.predict(np.asarray(x),np.asarray(x_g),np.asarray(x_u))           

            for k in range(j-i):        
                scipy.misc.imsave(self.args.result_dir+'/sr/'+str(count)+'.png', output[k])
                count=count+1
            i = j
	             
    def model_guidance_9(self, input_LR, input_g, reuse=False):
        with tf.variable_scope('generator', reuse):
            with tf.variable_scope('input_g_preprocessing'):
                with tf.variable_scope('conv1'):
                    x_g = utils.conv_layer(input_g, 3, 64, 1)
                    x_g = utils.relu_layer(x_g)
                    
                with tf.variable_scope('conv2'):
                    x_g = utils.conv_layer(x_g, 3, 64, 2)
                    x_g = utils.relu_layer(x_g)
                    
                with tf.variable_scope('conv3'):
                    x_g = utils.conv_layer(x_g, 3, 64, 1)
                    x_g = utils.relu_layer(x_g)

                with tf.variable_scope('conv4'):
                    x_g = utils.conv_layer(x_g, 3, 64, 2)
                    x_g = utils.relu_layer(x_g)  
                    
                with tf.variable_scope('conv5'):
                    x_g = utils.conv_layer(x_g, 3, 64, 1)
                    x_g = utils.relu_layer(x_g)

                with tf.variable_scope('conv6'):
                    x_g = utils.conv_layer(x_g, 3, 64, 2)
                    x_g = utils.relu_layer(x_g)
            
            with tf.variable_scope('conv1'):
                x = utils.conv_layer(input_LR,3,self.args.feature_size,1,False)
            with tf.variable_scope('conv1_g'):
                x_g = utils.conv_layer(x_g,3,self.args.feature_size,1,False)
            conv_1 = x
            
            with tf.variable_scope('resblocks'):  
                with tf.variable_scope('resblocks_1'):
                    for i in range(self.args.merge_resblock):
                        with tf.variable_scope('resblock{}'.format(i+1)):
                            x = utils.resBlock_EDSR(x,3,num_filters=self.args.feature_size,scale=self.args.scaling_factor)
                        with tf.variable_scope('resblock_g{}'.format(i+1)):
                            x_g = utils.resBlock_EDSR(x_g,3,num_filters=self.args.feature_size,scale=self.args.scaling_factor)
                    x = tf.concat([x, x_g], axis=3)
                    x = utils.conv_layer(x,3,self.args.feature_size,1,False)
                    
                with tf.variable_scope('resblocks_2'):
                    for i in range(self.args.merge_resblock):
                        with tf.variable_scope('resblock{}'.format(i+1)):
                            x = utils.resBlock_EDSR(x,3,num_filters=self.args.feature_size,scale=self.args.scaling_factor)
                        with tf.variable_scope('resblock_g{}'.format(i+1)):
                            x_g = utils.resBlock_EDSR(x_g,3,num_filters=self.args.feature_size,scale=self.args.scaling_factor)
                    x = tf.concat([x, x_g], axis=3)
                    x = utils.conv_layer(x,3,self.args.feature_size,1,False)
                                  
                with tf.variable_scope('resblocks_3'):
                    for i in range(self.args.merge_resblock):
                        with tf.variable_scope('resblock{}'.format(i+1)):
                            x = utils.resBlock_EDSR(x,3,num_filters=self.args.feature_size,scale=self.args.scaling_factor)
                        with tf.variable_scope('resblock_g{}'.format(i+1)):
                            x_g = utils.resBlock_EDSR(x_g,3,num_filters=self.args.feature_size,scale=self.args.scaling_factor)
                    x = tf.concat([x, x_g], axis=3)
                    x = utils.conv_layer(x,3,self.args.feature_size,1,False)
                      
                with tf.variable_scope('resblocks_4'):
                    for i in range(self.args.merge_resblock):
                        with tf.variable_scope('resblock{}'.format(i+1)):
                            x = utils.resBlock_EDSR(x,3,num_filters=self.args.feature_size,scale=self.args.scaling_factor)
                            
            with tf.variable_scope('conv_before_skip'):           
                x = utils.conv_layer(x,3,self.args.feature_size,1,False)
                
            with tf.variable_scope('upsampler_input'):
                x += conv_1   

            with tf.variable_scope('upsamplers'):
                for i in range(3):
                    with tf.variable_scope('upsampler{}'.format(i+1)):
                        x = utils.upsample_EDSR(x,2,self.args.feature_size,False)
        
            with tf.variable_scope('conv_final'):    
                x = utils.conv_layer(x,3,self.args.output_channels,1,False)
        self.g_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return x
   
    def warper_11(self, input_LR, input_g, is_training, reuse=False):
        with tf.variable_scope('warper', reuse=reuse):
            x = tf.concat([input_LR, input_g], axis=3)
            with tf.variable_scope('input_encoder'):
                with tf.variable_scope('conv1'):
                    x = utils.conv_layer(x, 3, 64, 1)
                    x = utils.relu_layer(x)
                    
                with tf.variable_scope('conv2'):
                    x = utils.conv_layer(x, 3, 64, 2)
                    x = utils.relu_layer(x)
                    
                with tf.variable_scope('conv3'):
                    x = utils.conv_layer(x, 3, 64, 1)
                    x = utils.relu_layer(x)

                with tf.variable_scope('conv4'):
                    x = utils.conv_layer(x, 3, 64, 2)
                    x = utils.relu_layer(x)    
                    
                with tf.variable_scope('conv5'):
                    x = utils.conv_layer(x, 3, 64, 1)
                    x = utils.relu_layer(x)

                with tf.variable_scope('conv6'):
                    x = utils.conv_layer(x, 3, 64, 2)
                    x = utils.relu_layer(x)

                with tf.variable_scope('conv_input_res'):
                    x = utils.conv_layer(x, 3, 64, 1, False)   
                    conv_1 = x
                with tf.variable_scope('resblocks'):
                    for i in range(8):
                        with tf.variable_scope('resblock{}'.format(i+1)):
                            x = utils.resBlock_EDSR(x, filter_size=3, num_filters=64)

                with tf.variable_scope('conv_before_skip'):
                    x = utils.conv_layer(x, 3, 64, 1, False)
                
                with tf.variable_scope('flow_decoder_input'):
                    x += conv_1
            
            with tf.variable_scope('flow_decoder'):
                with tf.variable_scope('upsampler1'):
                    x = utils.upsample_EDSR(x, 2, 64, False)
                    
                with tf.variable_scope('upsampler2'):
                    x = utils.upsample_EDSR(x, 2, 64, False)
                    
                with tf.variable_scope('upsampler3'):
                    x = utils.upsample_EDSR(x, 2, 64, False)
                    
                with tf.variable_scope('conv1'):
                    num_input_channels=x.get_shape().as_list()[3]
                    shape = [3, 3, num_input_channels, 2]
                    length=[2]
                    weights1 = tf.get_variable(name='weight', shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                    biases1 = tf.get_variable(name='bias', shape=length, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                    x = tf.nn.conv2d(x, weights1, [1, 1, 1, 1], padding='SAME')
                    x = tf.nn.bias_add(x, biases1)
         
        self.w_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='warper')
        return x 