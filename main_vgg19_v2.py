# -*- coding: utf-8 -*-
"""
Created on Wed May 20 04:12:40 2020

@author: USER
"""

# In[import]:
import tensorflow as tf
import os
import sys
import scipy.io
import time
import datetime
from IPython.display import Audio, display
import numpy as np
import librosa
import matplotlib.pyplot as plt
from barkmel import *
from utils import *
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('import,finish')

# In[filename get]:

#CONTENT_FILENAME = sys.argv[1]
#STYLE_FILENAME = sys.argv[2]

CONTENT_FILENAME = "audio_data/EG4.wav"
STYLE_FILENAME = "audio_data/Huang VC.wav"

#vol_balence(CONTENT_FILENAME,STYLE_FILENAME)

N_FFT = 4096
print('filename get,finish')
# In[get features]:

# Reads wav file and produces spectrum
# Fourier phases are ignored

a_content, fs ,p = read_audio_spectum(CONTENT_FILENAME)
a_style, fs ,_= read_audio_spectum(STYLE_FILENAME)

#content_vol=np.mean(a_content)
# shape[1]為x軸長度 [0]為y軸長度

N_SAMPLES = a_content.shape[1]
N_CHANNELS = a_content.shape[0]
a_style = a_style[:N_CHANNELS, :N_SAMPLES]

plt.figure(figsize=(9,9),dpi=120)
plt.subplot(1, 2, 1)
plt.title('Content')
plt.imshow(a_content[:400,:])
plt.subplot(1, 2, 2)
plt.title('Style')
plt.imshow(a_style[:400,:])
plt.show()

N_FILTERS = 2049

a_content_tf = np.ascontiguousarray(a_content.T[None,None,:,:])
a_style_tf = np.ascontiguousarray(a_style.T[None,None,:,:])

std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11)) #標準差
result_number=0

while result_number < 1 : 
    rn_S=str(result_number)
    time_now=datetime.datetime.now()
    time_str=time_now.strftime("%d-%m-%Y %H %M %S")
    ALPHA= 5e-2;
    if ALPHA == 5e-2:
        kernel = np.random.randn(1, 10, N_CHANNELS, N_FILTERS)*std   
        # "[濾波器長, 濾波器寬, 輸入頻道數, 輸出頻道數]"
        g = tf.Graph()
        with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
            x = tf.placeholder('float32', [1,1,N_SAMPLES,N_CHANNELS], name="x")
            # 資料形式[單次運算數, 資料長, 資料寬, 輸入頻道數]"
            layers=('con1_1','conv1_2','pool1','conv2_1','conv2_2','pool2',
                    'conv3_1','conv3_2','conv3_3','conv3_4','pool3'
                    'conv4_1','conv4_2','conv4_3','conv4_4','pool4'
                    'conv5_1','conv5_2','conv5_3','conv5_4','pool5')
            kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
            for i,name in enumerate(layers):                                          
                layer_type=name[:4]
                if layer_type == 'con1':
                    conv = tf.nn.conv2d(
                            x,
                            kernel_tf,
                            strides=[1, 1, 1, 1],
                            padding="SAME",
                            name="conv")
                    net= tf.nn.relu(conv)
                elif layer_type == 'conv':
                    net = tf.nn.conv2d(
                            net,
                            kernel_tf,
                            strides=[1, 1, 1, 1],
                            padding="SAME",
                            name="conv")
                    net= tf.nn.relu(conv)
                elif layer_type == 'pool':
                    net = tf.nn.max_pool(net,ksize=(1,2,1,1)
                                        ,strides=(1,1,1,1),padding='SAME')
           
            print(net)    
            
            content_features = net.eval(feed_dict={x: a_content_tf})
            content_features_a=np.squeeze(content_features)
            style_features = net.eval(feed_dict={x: a_style_tf})
            style_features_a=np.squeeze(style_features)
            style_re_features = np.reshape(style_features, (-1, N_FILTERS))
            content_re_features = np.reshape(content_features, (-1, N_FILTERS))
            style_gram = np.matmul(style_re_features.T, style_re_features) / N_SAMPLES
            content_gram = np.matmul(content_re_features.T, content_re_features) / N_SAMPLES
        
            plt.figure(figsize=(9,9),dpi=120)
            plt.subplot(1, 2, 1)
            plt.title('content_features')
            plt.imshow(content_features_a.T[:400,:])
            plt.figure(figsize=(9,9),dpi=120)
            plt.subplot(1, 2, 2)
            plt.title('style_gram')
            plt.imshow(style_gram.T[:400,:800])
            print('get features,finish')
            np.save('VGG-19 Features/NEW_style_features_vgg19',style_features)
            np.save('VGG-19 Features/NEW_content_features_vgg19',content_features)
            np.save('VGG-19 Features/NEW_style_gram_vgg19',style_gram)
            np.save('VGG-19 Features/NEW_content_gram_vgg19',content_gram)
            np.save('VGG-19 Features/NEW_kernel_vgg19',kernel)
    else:
    
        kernel = np.load('VGG-19 Features/NEW_kernel_vgg19.npy') 
        style_features=np.load('VGG-19 Features/NEW_style_features_vgg19.npy')
        content_features=np.load('VGG-19 Features/NEW_content_features_vgg19.npy')
        style_gram=np.load('VGG-19 Features/NEW_style_gram_vgg19.npy')
        content_gram=np.load('VGG-19 Features/NEW_content_gram_vgg19.npy')
        content_features_a=np.squeeze(content_features)
        style_features_a=np.squeeze(style_features)
#        style_re_features = np.reshape(style_features, (-1, N_FILTERS))
#        content_re_features = np.reshape(content_features, (-1, N_FILTERS))
        plt.figure(figsize=(9,9),dpi=200)
        plt.subplot(1, 1, 1)
        plt.title('content_gram')
        plt.imshow(content_features_a.T[:400,:])
        plt.figure(figsize=(9,9),dpi=200)
        plt.subplot(1, 2, 1)
        plt.title('style_gram')
        plt.imshow(style_gram.T[:400,:800])

# In[風格轉換output生成]:
#from sys import stder

    learning_rate= 1e-4
    iterations = 10000
    result = None
    
                
    with tf.Graph().as_default():
    
# Build graph with variable input
#     x = tf.Variable(np.zeros([1,1,N_SAMPLES,N_CHANNELS], dtype=np.float32), name="x")
        x = tf.Variable(np.random.randn(1,1,N_SAMPLES,N_CHANNELS).astype(np.float32)*1e-3, name="x")
        
        kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
        layers=('con1_1','conv1_2','pool1','conv2_1','conv2_2','pool2',
                'conv3_1','conv3_2','conv3_3','conv3_4','pool3'
                'conv4_1','conv4_2','conv4_3','conv4_4','pool4'
                'conv5_1','conv5_2','conv5_3','conv5_4','pool5')
        kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
        for i,name in enumerate(layers):                                          
            layer_type=name[:4]
            if layer_type == 'con1':
                conv = tf.nn.conv2d(
                        x,
                        kernel_tf,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                        name="conv")
                net= tf.nn.relu(conv)
            elif layer_type == 'conv':
                net = tf.nn.conv2d(
                        net,
                        kernel_tf,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                        name="conv")
                net= tf.nn.relu(conv)
            elif layer_type == 'pool':
                net = tf.nn.max_pool(net,ksize=(1,2,1,1)
                      ,strides=(1,1,1,1),padding='SAME')
#        conv = tf.nn.conv2d(
#                x,
#                kernel_tf,
#                strides=[1, 1, 1, 1],
#                padding="VALID",
#                name="conv") 
#        net = tf.nn.relu(conv)

        content_loss = ALPHA *2* tf.nn.l2_loss(
                net - content_features)

        style_loss = 0

        _, height, width, number = map(lambda i: i.value, net.get_shape())
#    size = height * width * number
        
        feats = tf.reshape(net, (-1, number))
        gram = tf.matmul(tf.transpose(feats), feats)  / N_SAMPLES
        style_loss = 2*tf.nn.l2_loss(gram - style_gram)

     # Overall loss
        loss = content_loss + style_loss
    
#Adam:train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss)
        opt = tf.contrib.opt.ScipyOptimizerInterface(loss, 
            method='L-BFGS-B', options={'maxiter': iterations})
        start = time.time()
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            print('Started optimization.')
            opt.minimize(sess)
#        for i in range(iterations):
#            config = tf.ConfigProto()
#            config.gpu_options.per_process_gpu_memory_fraction = 0.7
#            sess.run([train_op,loss,x])
#            if i % 100 ==0 and i >0:
#               
#                result_tmp = x.eval()
#                a = np.zeros_like(a_content)
#                a[:N_CHANNELS,:] = np.exp(result_tmp[0,0].T) - 1
#                xs=x
#                p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
#                for i2 in range(1000):
#                    S = a * np.exp(1j*p)
#                    xs = librosa.istft(S)
#                    p = np.angle(librosa.stft(xs, N_FFT))  
#                   
#                i_S=str(i)
#                A_S=str(ALPHA)
#               
#                OUTPUT_FILENAME = 'outputs/temp/temp'+ i_S+'.wav'
#                fig_address='outputs/temp/fig/'+ A_S+ '_'+ i_S+'.png'
#                librosa.output.write_wav(OUTPUT_FILENAME , xs, fs)
#                a_out, _ , _ = read_audio_spectum(OUTPUT_FILENAME)
#                a_out = a_out[:N_CHANNELS, :N_SAMPLES]
#               
#                plt.figure(figsize=(6,6),dpi=100)
#                plt.subplot(1, 2, 1)
#                plt.title('Content')
#                plt.imshow(a_content[:400,:])
#                plt.subplot(1, 2, 2)
#                plt.title('generate')
#                plt.imshow(a_out[:400,:])    
#                plt.savefig(fig_address)
#               
#                print('Epoch:%5d,loss:%.3f,style_loss:%.3f,content_loss:%.3f'
#                      %(i,loss.eval(),style_loss.eval(),content_loss.eval()))
#                print("time:{}".format(timeSince(start)))
#                del result_tmp, a_out, S, p, xs, i_S
            print ('Final loss:', loss.eval())
            result = x.eval()
    print("Finish lossFunction setting & Result generated")

# In[10]:


    a = np.zeros_like(a_content)
    a[:N_CHANNELS,:] = np.exp(result[0,0].T) - 1
    
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for i in range(1000):
        S = a * np.exp(1j*p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))

    x1,x2=make_stereo(x)
    OUTPUT_FILENAME = 'outputs/vgg19'+'_'+rn_S+'_'+time_str+'.wav'
    librosa.output.write_wav(OUTPUT_FILENAME , x2, fs)
    print (OUTPUT_FILENAME)


    a_out, fs , _ = read_audio_spectum(OUTPUT_FILENAME)
    a_out = a_out[:N_CHANNELS, :N_SAMPLES]
    fig_address='outputs/img/vgg19'+'_'+rn_S+'_'+time_str+'.png'
    plt.figure(figsize=(9,2),dpi=200)
    plt.subplot(1, 3, 1)
    plt.title('Content')
    plt.imshow(a_content[:400,:])
    plt.subplot(1, 3, 2)
    plt.title('Style')
    plt.imshow(a_style[:400,:])
    plt.subplot(1, 3, 3)
    plt.title('Generated')
    plt.imshow(a_out[:400,:])    
    plt.savefig(fig_address)
    plt.show()
    # vol_balence(CONTENT_FILENAME,OUTPUT_FILENAME) 
    result_number=result_number + 1


