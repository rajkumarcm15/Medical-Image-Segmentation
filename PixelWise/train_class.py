import numpy as np
import Data as data
import evaluate_print as ev
import threading
import gc
import StatusSaver as log
import tensorflow as tf

log = log.StatusSaver
def init_weight(shape):
  initial = tf.truncated_normal(shape,stddev=0.1)
  return tf.Variable(initial)

def init_bias(shape):
  initial = tf.constant(0.1,shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def conduct_test(test1,test2,test3,i):
    if not ( test1 and test2 and test3 ):
        test1 = data.Data(3,dir_path,
            	            p_shape=[_x_,_y_],n_samples=n_patches,training=False)
        test2 = data.Data(8,dir_path,
            	            p_shape=[_x_,_y_],n_samples=n_patches,training=False)
        test3 = data.Data(0,dir_path,
            	            p_shape=[_x_,_y_],n_samples=n_patches,training=False)

    test_data1, test_targ1,_ = test1.get_next()
    test_data2, test_targ2,_ = test2.get_next()
    test_data3, test_targ3,_ = test3.get_next()
    test_data = np.concatenate([test_data1,test_data2,test_data3])
    test_targ = np.concatenate([test_targ1,test_targ2,test_targ3])
    
    test_data = sess.run(tf.reshape(test_data,[-1,_x_,_y_,1]))
    test_targ = sess.run(tf.reshape(test_targ,[-1,3]))

    pred = sess.run(y_pred,feed_dict={x:test_data,y_:test_targ,keep_prob:1})
    
    log("status.txt",["Validation Results"])
    thread = threading.Thread(target=ev.print_cscore,args=(i,pred,test_targ))
    thread.daemon = True
    thread.start()

def init_data():
    data1 = data.Data(3,dir_path,
                        p_shape=[_x_,_y_],n_samples=n_patches,training=True)
    data2 = data.Data(8,dir_path,
                        p_shape=[_x_,_y_],n_samples=n_patches,training=True)
    data3 = data.Data(0,dir_path,
                        p_shape=[_x_,_y_],n_samples=n_patches,training=True)
    return data1, data2, data3

def save_status(epoch,g_c,l_c):
    f = open('status.txt','w')
    f.truncate()
    s = ("epoch: %d, global count: %d, local count: %d"%(epoch,global_count,local_count))
    # print(s)
    f.write(s)
    f.close()

n_aug = 3
n_patches = 3
_x_ = 112
_y_ = 112
offset = 0.1
scale = 1
depth = 1
N_CLASS = 3
# dir_path ='/vol/medic01/users/mrajchl/data/AbdominalCT'
dir_path ='/Users/Rajkumar/Documents/ISO/CNN/Project/Data'

test1 = None
test2 = None
test3 = None

batch = ( n_patches * n_aug )  * N_CLASS
x = tf.placeholder("float",shape=[batch,_x_,_y_,depth])
y_ = tf.placeholder("float",shape=[batch,N_CLASS])

# Conv Block1

# 112x112
W_conv1 = init_weight([5,5,depth,32])
b_conv1 = init_bias([32])
in_conv1 = conv2d(x,W_conv1)+b_conv1
mew1,var1 = tf.nn.moments(in_conv1, axes=[0],keep_dims=False)
bn_conv1 = tf.nn.batch_normalization(in_conv1,mew1,var1,offset,scale,1e-5)
h_conv1 = tf.nn.relu(bn_conv1)

W_conv2 = init_weight([5,5,32,32])
b_conv2 = init_bias([32])
in_conv2 = conv2d(h_conv1,W_conv2)+b_conv2
mew1,var1  = tf.nn.moments(in_conv2, axes=[0],keep_dims=False)
bn_conv2 = tf.nn.batch_normalization(in_conv2,mew1,var1,offset,scale,1e-5)
h_conv2 = tf.nn.relu(bn_conv2)

h_pool1 = max_pool_2x2(h_conv2)

# Conv Block2
# 56x56
W_conv3 = init_weight([3,3,32,64])
b_conv3 = init_bias([64])
in_conv3 = conv2d(h_pool1,W_conv3)+b_conv3
mew1,var1 = tf.nn.moments(in_conv3,axes=[0],keep_dims=False)
bn_conv3 = tf.nn.batch_normalization(in_conv3,mew1,var1,offset,scale,1e-5)
h_conv3 = tf.nn.relu(bn_conv3)

W_conv4 = init_weight([3,3,64,64])
b_conv4 = init_bias([64])
in_conv4 = conv2d(h_conv3,W_conv4)+b_conv4
mew1,var1 = tf.nn.moments(in_conv4,axes=[0],keep_dims=False)
bn_conv4 = tf.nn.batch_normalization(in_conv4,mew1,var1,offset,scale,1e-5)
h_conv4 = tf.nn.relu(bn_conv4)

h_pool2 = max_pool_2x2(h_conv4)

# Conv Block3
# 28x28
W_conv5 = init_weight([3,3,64,128])
b_conv5 = init_bias([128])
in_conv5 = conv2d(h_pool2,W_conv5)+b_conv5
mew1,var1 = tf.nn.moments(in_conv5,axes=[0],keep_dims=False)
bn_conv5 = tf.nn.batch_normalization(in_conv5,mew1,var1,offset,scale,1e-5)
h_conv5 = tf.nn.relu(bn_conv5)

W_conv6 = init_weight([3,3,128,128])
b_conv6 = init_bias([128])
in_conv6 = conv2d(h_conv5,W_conv6)+b_conv6
mew1,var1 = tf.nn.moments(in_conv6,axes=[0],keep_dims=False)
bn_conv6 = tf.nn.batch_normalization(in_conv6,mew1,var1,offset,scale,1e-5)
h_conv6 = tf.nn.relu(bn_conv6)

h_pool3 = max_pool_2x2(h_conv6)

#-------------------------------------------------------------------------------------------#*
#---------Convolution layer for classification ---------------------------------------------#*
#-------------------------------------------------------------------------------------------#*
                                                                                            #*
# 7x7x512       FULLY CONNECTED 1                                                           #*
# 3 classes -> FG - Kidney, Liv, and BG                                                     #*
W_fc1 = init_weight([14*14*128,500])                                                   #*
b_fc1 = init_bias([500])                                                                   #*
fc1_input = tf.reshape(h_pool3,[-1,14*14*128])
in_fc1 = tf.matmul(fc1_input,W_fc1)+b_fc1
mew1,var1 = tf.nn.moments(in_fc1,axes=[0],keep_dims=True)
bn_fc1 = tf.nn.batch_normalization(in_fc1,mew1,var1,offset,scale,1e-5)
h_fc1 = tf.nn.relu(bn_fc1)                                        #*
#  Output: 600x1x1x1024                                                                     #*
#-----------------------Dropout---------------------------------------------------------------

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#--------------------------------------------------------------------------------------------#*
#--------------------------Calculating Dropout in Readout layer------------------------------#*
#--------------------------------------------------------------------------------------------#*
W_fc2 = init_weight([500,3])
b_fc2 = init_bias([3])
in_fc2 = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
mew1,var1 = tf.nn.moments(in_fc2,axes=[0],keep_dims=True)
bn_fc2 = tf.nn.batch_normalization(in_fc2,mew1,var1,offset,scale,1e-5)
y_pred = tf.nn.softmax(bn_fc2)
#-------------------------------------------------------------------------------------------#*
#-----------------------Evaluation----------------------------------------------------------#*

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_pred,y_)
train_step = tf.train.MomentumOptimizer(1e-4,0.9).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
sess = tf.Session()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
data1, data2, data3 = init_data()

global_count = 0
local_count = 0
for i in range(50):
  log("status.txt",["epoch: %d"%(i), "global count: %d"%(global_count),"local count: 0"])
  while True:
      train_data1, train_targ1, c1 = data1.get_next()
      train_data2, train_targ2, c2 = data2.get_next()
      train_data3, train_targ3, c3 = data3.get_next()
      if not ( c1 and c2 and c3 ):
          train_data = np.concatenate([train_data1,train_data2,train_data3])
          train_targ = np.concatenate([train_targ1,train_targ2,train_targ3])
          train_data = sess.run(tf.reshape(train_data,[-1,_x_,_y_,1]))
          train_targ = sess.run(tf.reshape(train_targ,[-1,1,3,1]))

          bundle = zip(train_data,train_targ)
          np.random.shuffle(bundle)
          train_data, train_targ = zip(*bundle)
          train_data = np.concatenate(train_data).reshape(-1,_x_,_y_,1)
          train_targ = np.concatenate(train_targ).reshape(-1,3)

          if (global_count)%50 == 0 and local_count != 0:
            log("status.txt",["global count %d"%(global_count),"local count %d"%(local_count)])
            obtained_pred = sess.run(y_pred,feed_dict={x:train_data,y_:train_targ,keep_prob:0.5})
            thread = threading.Thread(target=ev.print_cscore, args=(i,obtained_pred,train_targ))
            thread.daemon = True
            thread.start()
            conduct_test(test1,test2,test3,global_count)

          if i%20 == 0 and i != 0:
            save_path = saver.save(sess, "model%d.ckpt"%(i))

          sess.run(train_step,feed_dict={ x: train_data, y_: train_targ, keep_prob:0.5})
          global_count += 1
          local_count += 1
      else:
          local_count = 0
          data1, data2, data3 = init_data()
          gc.collect()
          break


#--------------------------- Testing Phase ------------------------------------------------------------#


save_path = saver.save(sess, "model_final.ckpt")
conduct_test(test1,test2,test3,50)
