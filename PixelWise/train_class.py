import numpy as np
import Data as data
import evaluate_print as ev
import threading
import StatusSaver as log
import tensorflow as tf
import multiprocessing
# from memory_profiler import profile

log = log.StatusSaver


def init_weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="W")


def init_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="B")


def conv3d(x, W, padding='SAME'):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding=padding, name="conv3d")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool")


def conduct_test(test1, test2, i):
    if not (test1 and test2):
        test1 = data.Data(3, dir_path,
                          p_shape=[_x_, _y_, _z_], n_samples=n_patches, training=False)
        test2 = data.Data(8, dir_path,
                          p_shape=[_x_, _y_, _z_], n_samples=n_patches, training=False)
        # test3 = data.Data(0,dir_path,
        #     	            p_shape=[_x_,_y_],n_samples=n_patches,training=False)

    test_data1, test_targ1, _ = test1.get_next()
    test_data2, test_targ2, _ = test2.get_next()
    # test_data3, test_targ3,_ = test3.get_next()
    test_data = np.concatenate([test_data1, test_data2])
    test_targ = np.concatenate([test_targ1, test_targ2])

    bundle = zip(test_data, test_targ)
    np.random.shuffle(bundle)
    test_data, test_targ = zip(*bundle)
    test_data = np.concatenate(test_data).reshape([-1, 1, _x_, _y_, _z_])
    test_targ = np.concatenate(test_targ).reshape([-1, 3])

    pred = sess.run(y_pred, feed_dict={x: test_data, y_: test_targ, keep_prob: prob})

    log("status.txt", ["Validation Results"])
    thread = threading.Thread(target=ev.print_cscore, args=(i, pred, test_targ))
    thread.daemon = True
    thread.start()


def init_data():
    data1 = data.Data(3, dir_path,
                      p_shape=[_x_, _y_, _z_], n_samples=n_patches, training=True)
    data2 = data.Data(8, dir_path,
                      p_shape=[_x_, _y_, _z_], n_samples=n_patches, training=True)
    # data3 = data.Data(0,dir_path,
    #                     p_shape=[_x_,_y_],n_samples=n_patches,training=True)
    return data1, data2  # , data3


def save_status(epoch, g_c, l_c):
    f = open('status.txt', 'w')
    f.truncate()
    s = ("epoch: %d, global count: %d, local count: %d" % (epoch, global_count, local_count))
    # print(s)
    f.write(s)
    f.close()


n_aug = 2
n_patches = 1
_x_ = 150
_y_ = 150
_z_ = 1
offset = 0.1
scale = 1
depth = 1
channels = _z_
N_CLASS = 2
prob = 0.6
# dir_path ='/vol/medic01/users/mrajchl/data/AbdominalCT'
dir_path = '/Users/Rajkumar/ISO/CNN/Project/Data'

test1 = None
test2 = None
# test3 = None

pos_examples = int(np.ceil(n_patches * 1.0 / 2))  # 1
pos_examples += (pos_examples * 2) # 1 + 2 = 3
neg_examples = int(np.floor(n_patches * 1.0 / 2))
batch = (pos_examples + neg_examples) * N_CLASS



print("Defining graph")
#@profile
def add_conv_block(x,add_nonlinear,output_channels,receptive_size=5,name="conv"):
    with tf.name_scope(name):
        channels = x._shape[4].value
        W_conv1 = init_weight([1, receptive_size, receptive_size, channels, output_channels])
        b_conv1 = init_bias([output_channels])
        mew1, var1 = tf.nn.moments(x, axes=[0, 1], keep_dims=False)
        x = tf.nn.batch_normalization(x, mew1, var1, offset, scale, 1e-5,name="BN")
        h_conv1 = conv3d(x, W_conv1) + b_conv1
        del x
        h_conv1 = tf.nn.relu(h_conv1)
        tf.summary.histogram("weights",W_conv1)
        tf.summary.histogram("biases",b_conv1)
        tf.summary.histogram("activations",h_conv1)

        # Non-linear conv block with same output channels
        if add_nonlinear:
            h_conv1 = add_conv_block(h_conv1,False,output_channels)
            h_pool1 = max_pool_2x2(h_conv1[:, 0, :, :, :])
            h_pool1 = tf.reshape(h_pool1, [h_pool1._shape[0].value, 1, h_pool1._shape[1].value,
                                           h_pool1._shape[2].value, h_pool1._shape[3].value])
            return h_pool1
        else:
            return h_conv1

#@profile
def add_fcl(x,keep_prob,name="fully_convolution"):
    with tf.name_scope(name):
        r_size = x._shape[2].value
        channels = x._shape[4].value
        W_fc1 = init_weight([1, r_size, r_size, channels, 128])
        b_fc1 = init_bias([128])
        mew1, var1 = tf.nn.moments(x, axes=[0, 1], keep_dims=True)
        x = tf.nn.batch_normalization(x, mew1, var1, 0.1, 1, 1e-6,name="BN")
        h_fc1 = conv3d(x, W_fc1, 'VALID') + b_fc1
        h_fc1 = tf.nn.relu(h_fc1)
        tf.summary.histogram("weights",W_fc1)
        tf.summary.histogram("biases",b_fc1)
        tf.summary.histogram("activations",h_fc1)


    # --------------------------------------------------------------------------------------------#*
    # --------------------------Calculating Dropout in Readout layer------------------------------#*
    # --------------------------------------------------------------------------------------------#*
    with tf.name_scope("fc"):
        W_fc2 = init_weight([1 * 128, 3])
        b_fc2 = init_bias([3])
        in_fc2 = tf.reshape(h_fc1, [-1, 128])
        in_fc2 = tf.nn.dropout(in_fc2, keep_prob, name="dropout")
        h_fc2 = tf.matmul(in_fc2, W_fc2) + b_fc2
        y_pred = tf.nn.softmax(h_fc2)

        tf.summary.histogram("weights",W_fc2)
        tf.summary.histogram("biases",b_fc2)
        tf.summary.histogram("activations",y_pred)
        return h_fc2,y_pred


def get_cross_entropy(x,y):
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(x, y),
                                       name="cross_entropy") # y_
        tf.summary.scalar("cross entropy",cross_entropy)
        return cross_entropy

def train(cross_entropy):
    with tf.name_scope("train"):
        train_step = tf.train.MomentumOptimizer(1e-6, 0.99).minimize(cross_entropy)
        return train_step


NUM_CORES = multiprocessing.cpu_count() * 2
config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                        intra_op_parallelism_threads=NUM_CORES)
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=[batch, depth, _x_, _y_, channels])
    y_ = tf.placeholder(dtype=tf.float32, shape=[batch, N_CLASS + 1])
    with tf.Session(config=config) as sess:
        # Define graph
        for i in range(3):
            pool = add_conv_block(x,True,int(2**(5+i)))
        keep_prob = tf.placeholder("float")
        h_fc2,y_pred = add_fcl(pool,keep_prob)
        cross_entropy = get_cross_entropy(h_fc2,y_)
        train_step = train(cross_entropy)
        sess.run(tf.global_variables_initializer())

        summ = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/tmp/conv_net")
        writer.add_graph(sess.graph)

        saver = tf.train.Saver()
        # saver.restore(sess,"backup.cpkt")
        data1, data2 = init_data()

        global_count = 0
        local_count = 0
        print("Graph created")
        err_rts = []
        for i in range(10):
            log("status.txt", ["epoch: %d" % (i), "global count: %d" % (global_count), "local count: 0"])

            while True:
                train_data1, train_targ1, c1 = data1.get_next()
                train_data2, train_targ2, c2 = data2.get_next()

                if not (c1 and c2):
                    train_data = np.concatenate([train_data1, train_data2])
                    train_targ = np.concatenate([train_targ1, train_targ2])
                    # train_data = sess.run(tf.reshape(train_data, [-1, 1, _x_, _y_, _z_]))
                    train_targ = sess.run(tf.reshape(train_targ, [-1, 1, 3, 1]))

                    bundle = zip(train_data, train_targ)
                    np.random.shuffle(bundle)
                    train_data, train_targ = zip(*bundle)
                    train_data = np.concatenate(train_data).reshape([-1, 1, _x_, _y_, _z_])
                    train_targ = np.concatenate(train_targ).reshape([-1, 3])

                    if (global_count) % 20 == 0 and local_count != 0:
                        log("status.txt", ["global count %d" % (global_count), "local count %d" % (local_count)])
                        obtained_pred = sess.run(y_pred, feed_dict={x: train_data, y_: train_targ, keep_prob: prob})
                        thread = threading.Thread(target=ev.print_cscore, args=(i, obtained_pred, train_targ))
                        thread.daemon = True
                        thread.start()
                        conduct_test(test1, test2, global_count)

                        # if i%20 == 0 and i != 0:
                    #                 save_path = saver.save(sess, "model%d.ckpt"%(i))

                    err, _ = sess.run([cross_entropy, train_step],
                                      feed_dict={x: train_data, y_: train_targ, keep_prob: prob})
                    err_rts.append(np.mean(err))

                    if local_count % 10 == 0:
                        print("Local count: %d" % local_count)
                        print("Error rate: ", np.mean(err))

                    global_count += 1
                    local_count += 1

                else:
                    local_count = 0
                    data1, data2 = init_data()
                    break

        # --------------------------- Testing Phase ------------------------------------------------------------#


        save_path = saver.save(sess, "model_final.ckpt")
        conduct_test(test1, test2, 50)
