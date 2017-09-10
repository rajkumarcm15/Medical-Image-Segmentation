"""
 Name: Rajkumar Conjeevaram Mohan
 Project: Medical Image Segmentation and Reconstruction using Deep Learning
 This file involves loading, and preprocessing of medical
 image data required for the training of Convolutional Neural Network
"""

# This file assumes medpy, numpy and their contents are already imported by
# the class that invokes this.

from os import listdir
from os.path import join, isfile
# from numpy import mean,std,tile,random
from medpy.io import load
import numpy as np
import time
import threading
import logging
# from memory_profiler import profile

class Data:
    __author__ = 'Rajkumar Conjeevaram Mohan'
    b_pointer = 0
    dim_pointer = -1
    dir_path = ""
    file_names = []
    LIVER = 3
    KIDNEY = 8
    buffer_filled = False
    train_data = None
    train_targ = None
    temp_img = None
    temp_lbl = None
    complete = False

    def __init__(self,
                 id,
                 dir_path,
                 p_shape,
                 n_samples = 1,
                 numeric_targ = False,
                 augment_sigma = 0.1,
                 training = True):

        self.dir_path = dir_path
        dir_path += "/aff_imgs"
        self.n_samples = n_samples
        self.id = id
        self.neg_id = 0 #CHANGE THIS TO DYNAMIC
        self.file_names_main = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        self.TEST_INDEX = 100
        self.file_names = []
        if training:
            self.file_names = self.file_names_main[:self.TEST_INDEX]
            # print self.file_names
        else:
            self.file_names = self.file_names_main[self.TEST_INDEX:]

        n_data = len(self.file_names)
        self.indices = np.random.permutation(n_data)[:50]
        self.p_shape = p_shape
        self.augment_sigma = augment_sigma
        self.numeric_targ = numeric_targ
        self.log = logging.getLogger()
        # self.get_next()
        self.x_rad = p_shape[0] / 2
        self.y_rad = p_shape[1] / 2
        self.z_rad = int(np.ceil(p_shape[2] * 1.0 / 2))
        self.thread_start()

    # UPDATE THIS
    def get_next(self):
        N_Z = 371
        train_data = None
        train_targ = None

        while not self.buffer_filled:
            # print "Waiting for the buffer to be filled"
            time.sleep(1)

        self.buffer_filled = False

        train_data = self.train_data
        train_targ = self.train_targ

        self.thread_start()

        return train_data, train_targ, self.complete
        # else:
        #     return self

    def get_test(self):
        imgs = []
        lbls = []
        fns = self.file_names_main[self.TEST_INDEX:]
        for i in range(50):
            X_fn = self.dir_path+"/aff_imgs/"+fns[i]
            Y_fn = self.dir_path+"/aff_lbls/"+fns[i]
            img = self.load_data(X_fn)
            lbl = self.load_data(Y_fn)
            b = img.shape[2]
            x = img.shape[0]
            y = img.shape[1]
            img_s = np.reshape(np.concatenate(np.dsplit(img,2)),[b,x,y,1])
            lbl_s = np.reshape(np.concatenate(np.dsplit(lbl,2)),[b,x,y,1])
            imgs.append(img_s)
            lbls.append(lbl_s)
        imgs = np.concatenate(imgs)
        lbls = np.concatenate(lbls)
        return imgs, lbls


    def thread_start(self):
        thread = threading.Thread(target=self.process_data,args=(self.id,self.p_shape))
        # thread = threading.Thread(target=self.process_data(self.id,self.p_shape))
        thread.daemon = True
        thread.start()

    def refine_pos(self,nx,ny,nz,pos):
        pos = zip(pos[0],pos[1],pos[2])

        # positions = filter(lambda x: (x[0]+x_rad <= nx and x[0]-x_rad >= 0 and
        #                               ny <= x[1]+y_rad and x[1]-y_rad >= 0), block)
        try:
            pos = np.matrix(pos)
            a = np.where(pos[:,0]+self.x_rad <= nx)
            b = np.where(pos[:,0]-self.x_rad >= 0)
            c = np.where(pos[:,1]+self.y_rad <= ny)
            d = np.where(pos[:,1]-self.y_rad >= 0)
        except:
            print("Exception thrown at refine_pos")
        # e = np.where(pos[:,2]+self.z_rad <= nz)
        # f = np.where(pos[:,2]-self.z_rad >= 0)

        indices = np.intersect1d(a[0],np.intersect1d(b[0],np.intersect1d(c[0],d[0])))
        return np.asarray(pos[indices,:])
        # return positions

    def get_z_min_max(self,s_pos,p_shape):
        z_min = s_pos[2] - p_shape[2]
        if z_min < 0:
            z_min = 0
        if (s_pos[2] - z_min) == p_shape[2]:
            z_max = s_pos[2]
        else:
            left_over = p_shape[2] - (s_pos[2] - z_min)
            z_max = s_pos[2] + left_over
        return z_min,z_max

    # @profile
    def process_data(self,id,p_shape):
        imgs = []
        # s_c -> samples computed
        s_c = 0
        offsets = []
        imset = []
        n_pos_samples = int(np.ceil(self.n_samples*1.0/2))
        min_threshold = 900
        n_neg_samples = int(np.floor(self.n_samples/2))
        while True:
            b_pointer = self.b_pointer
            # b_pointer = self.shift_pointer(N_Z,b_pointer_old)

            X_fn = self.dir_path+"/aff_imgs/"+self.file_names[self.indices[b_pointer]]
            Y_fn = self.dir_path+"/aff_lbls/"+self.file_names[self.indices[b_pointer]]
            img = self.load_data(X_fn)
            lbl = self.load_data(Y_fn)
            nx = img.shape[0]
            ny = img.shape[1]
            nz = img.shape[2]

            # Get the positions of voxels where annotations are located
            pos = np.where(lbl == id)

            #-----------------------------------------------------------------------------------------
            # SELECT A RANDOM COORDINATE THAT MEETS THE REQUIREMENT-----------------------------------
            #-----------------------------------------------------------------------------------------
            rand_pos_found = False
            count = 0
            s_pos = []
            if len(pos[0]) > min_threshold:
                while not rand_pos_found and count < len(pos[0]):
                    rand_i = np.random.randint(0,len(pos[0]))
                    # s_pos -> sample position
                    s_pos = np.array([pos[0][rand_i],pos[1][rand_i],pos[2][rand_i]])
                    count += 1
                    if (s_pos[0] - self.x_rad) > 0 and (s_pos[0] + self.x_rad) <= nx and \
                        (s_pos[1] - self.y_rad) > 0 and (s_pos[0] + self.y_rad) <= ny:
                        rand_pos_found = True
                    else:
                        s_pos = []
                        continue
            else:
                if len(pos[0]) == 0:
                    self.b_pointer += 1
                    continue
                else:
                    pos = self.refine_pos(nx, ny, nz, pos)
                    s_pos = pos[0]
            #-----------------------------------------------------------------------------------------

            # -----------------------------------------------------------------------------------------
            # UPDATE THE FILE POINTER SINCE A COORDINATE IS FOUND IN THIS IMAGE
            # -----------------------------------------------------------------------------------------
            if b_pointer == len(self.indices)-1:
                self.b_pointer = 0
                self.complete = True
            else:
                self.b_pointer += 1

            if len(pos[0]) != 0:
                x_min = s_pos[0]-self.x_rad
                x_max = s_pos[0]+self.x_rad
                y_min = s_pos[1]-self.y_rad
                y_max = s_pos[1]+self.y_rad

                # Lets handle the third dimension slightly differently since
                # it was found empirically that the third dimension is at most
                # of the times does not have even space i.e., radius before and
                # after its coordinate

                # z_max = nz
                # deviation = (s_pos[2] + p_shape[2]) - nz
                # z_min = s_pos[2]
                # if x_max > 0:
                #     z_max = s_pos[2] + p_shape[2] - deviation
                #     z_min = s_pos[2] - deviation

                # Assume s_pos[2] = 50, we use up the entire
                # space to its left
                # z_min = s_pos[2] - p_shape[2]
                # if z_min < 0:
                #     z_min = 0
                # if (s_pos[2] - z_min) == p_shape[2]:
                #     z_max = s_pos[2]
                # else:
                #     left_over = p_shape[2] - (s_pos[2]-z_min)
                #     z_max = s_pos[2] + left_over
                z_min,z_max = self.get_z_min_max(s_pos,p_shape)

                # Instead of looking for refining coordinates from large search space
                # we will first reduce our refinement to positions in reduced search
                # space (sample_patch)
                sample_patch = lbl[x_min:x_max,y_min:y_max,z_min:z_max]

                # Update the positions so that refine_pos performs the search for constrained
                # indices on small space rather than on entire image, which is computationally expensive
                pos = np.where(sample_patch == id)

                # The underlying code is commented since we no longer manually add
                # a static 3rd dimension to positions instead var `pos` by now
                # would automatically have 3rd dimension coordinates generated automatically
                # pos = list([pos[0],pos[1],np.tile([s_pos[2]],[pos[0].shape[0]])])

                pos = list([pos[0],pos[1],pos[2]])
                pos[0] += x_min
                pos[1] += y_min
                pos[2] += z_min
                positions = self.refine_pos(nx,ny,nz,pos)
                # Sample a 3D patch from the image
                # for i in range(self.n_samples):
                if (s_c + len(positions)) <= n_pos_samples:
                    s_c += len(positions)
                    offsets.append(positions)
                else:
                    size = n_pos_samples - s_c
                    indices = np.random.randint(0, len(positions), size=[size])
                    positions = positions[indices]
                    s_c = n_pos_samples
                    offsets.append(positions)
                # Each 'VOXEL' from the above routine would yield a complete
                # 3D image. Each depth slice is not counted as a separate sample
                # rather a voxel position...

                for i in range(len(offsets)):
                    patches = self.extract_patches(img, offsets[i], p_shape,nz)
                    patches,aug_patches = self.augment_data(patches)
                    patches = np.reshape(np.concatenate(patches,axis=0),[len(patches),p_shape[0],p_shape[1],-1])
                    aug_patches = np.concatenate(aug_patches,axis=0).reshape([len(aug_patches),p_shape[0],p_shape[1],-1])
                    imset.append(np.concatenate([patches,aug_patches],axis=0))
                    del patches,aug_patches
                    # For negative sampling
                    if n_neg_samples > 0:
                        for os in positions:
                            # print(self.id," - ",os)
                            # Offsets of positive class annotations are used since
                            # we want negative patch from an image with positive annotation
                            # not a blank label where np.sum(lbl) == 0
                            z_min,z_max = self.get_z_min_max(os,p_shape)
                            patch = lbl[os[0]-self.x_rad:os[0]+self.x_rad,os[1]-self.y_rad:os[1]+self.y_rad,
                                    z_min:z_max]
                            neg_pos = np.where(patch == self.neg_id)
                            neg_pos = self.refine_pos(nx,ny,nz,neg_pos)
                            try:
                                index = np.random.randint(0,len(neg_pos))
                            except Exception,e:
                                print("Error thrown at negative sampling: ",str(e))
                            neg_pos = neg_pos[index]
                            z_max = nz
                            deviation = (neg_pos[2] + p_shape[2]) - nz
                            z_min = neg_pos[2]
                            if x_max > 0:
                                z_max = neg_pos[2] + p_shape[2] - deviation
                                z_min = neg_pos[2] - deviation
                            temp = img[neg_pos[0]-self.x_rad:neg_pos[0]+self.x_rad,
                                       neg_pos[1]-self.y_rad:neg_pos[1]+self.y_rad,z_min:z_max]
                            temp = np.reshape(temp,[1,p_shape[0],p_shape[1],p_shape[2]])
                            imset.append(temp)
                            del patch, neg_pos, temp

                if s_c == n_pos_samples:
                    break
            else:
                continue

        del img # This is no longer needed; deleting to save some space
        # The while loop would be finished here

        temp_lbl = [0,0,0]
        if id == 3:
            temp_lbl[0] = 1
        elif id == 8:
            temp_lbl[1] = 1
        else:
            temp_lbl[2] = 1

        imset = np.concatenate(imset,axis=0)
        _b_, _x_, _y_, _z_ = imset.shape
        imset = np.reshape(imset, [_b_, 1, _x_, _y_, _z_])
        n_aug = 2
        lblset = np.tile(temp_lbl,[int(np.ceil(self.n_samples*1.0/2))+n_aug,1])
        lblset = np.concatenate([lblset,np.tile([0,0,1],[int(np.floor(self.n_samples*1.0/2)),1])],axis=0)

        self.train_data = imset
        self.train_targ = lblset
        del imset, lblset

        self.buffer_filled = True

    def load_data(self,file_name):
        epi_img,_ = load(file_name)
        return epi_img

    def extract_patches(self,img,offsets,p_shape,nz):
        im_set = []
        p_radius = [x/2 for x in p_shape]
        for i in range(len(offsets)):
            _os = offsets[i]
            x_min = _os[0] - p_radius[0]
            x_max = _os[0] + p_radius[0]
            y_min = _os[1] - p_radius[1]
            y_max = _os[1] + p_radius[1]
            # z_max = nz
            # deviation = (_os[2] + p_shape[2]) - nz
            # z_min = _os[2]
            # if x_max > 0:
            #     z_max = _os[2] + p_shape[2] - deviation
            #     z_min = _os[2] - deviation
            z_min,z_max = self.get_z_min_max(_os,p_shape)
            temp_img = img[x_min:x_max, y_min:y_max, z_min:z_max].astype('float64')
            # temp_img = np.reshape(temp_img,[x_max-x_min,y_max-y_min,z_max-z_min])
            im_set.append(temp_img)
            del temp_img
        return im_set

    def augment_data(self,imgs):
        im_set = []
        for i in range(len(imgs)):
            imgs[i] += np.random.normal(loc=0,scale=self.augment_sigma,size=imgs[i].shape)
            temp_im = imgs[i].astype('float64')
            nx = temp_im.shape[0]
            ny = temp_im.shape[1]
            nz = temp_im.shape[2]
            # Vertical Flipping
            flipp_v = np.zeros(temp_im.shape)

            for z in range(temp_im.shape[2]):
                temp_vert_flip = []
                for i in reversed(temp_im[:,:,z]):
                    temp_vert_flip.append(i)
                temp_vert_flip = np.concatenate(temp_vert_flip,axis=0)
                temp_vert_flip = np.reshape(temp_vert_flip,[nx,ny])
                flipp_v[:,:,z] = temp_vert_flip

            # Horizontal Flipping
            flipp_h = np.zeros(temp_im.shape)
            for z in range(temp_im.shape[2]):
                temp_hor_flip = []
                for i in reversed(temp_im[:,:,z].T):
                    temp_hor_flip.append(i)
                temp_hor_flip = np.concatenate(temp_hor_flip,axis=0)
                temp_hor_flip = np.reshape(temp_hor_flip,[nx,ny])
                flipp_h[:,:,z] = temp_hor_flip.T

            im_set.append(flipp_v)
            im_set.append(flipp_h)
        return imgs,im_set


####
# import matplotlib.pyplot as plt
# import tensorflow as tf
# sess = tf.Session()
# dir = '/Users/Rajkumar/ISO/CNN/Project/Data'
# _data1 = Data(3,dir,[150,150,110],numeric_targ=True,n_samples=1,training=True)
# _data2 = Data(8,dir,[150,150,371],numeric_targ=True,n_samples=2,training=True)

# train_data1, train_targ1, c1 = _data1.get_next()
# np.save('/Users/Rajkumar/Desktop/train_data1',train_data1)
# train_data2, train_targ2, c2 = _data2.get_next()
# train_data1 = tf.cast(train_data1,tf.float32)
# train_data2 = tf.cast(train_data2,tf.float32)
# # Normalisation
# mew1,var1 = tf.nn.moments(train_data1,axes=[0,1,2],keep_dims=False)
# train_data1 = tf.nn.batch_normalization(train_data1,mew1,var1,0.1,1,1e-5)
# mew2,var2 = tf.nn.moments(train_data2,axes=[0,1,2],keep_dims=False)
# train_data2 = tf.nn.batch_normalization(train_data2,mew2,var2,0.1,1,1e-5)
# train_data1,train_data2 = sess.run([train_data1,train_data2])
# train_data1 = sess.run([train_data1])
#  for i in range(10):
#    plt.figure()
#    plt.title("Liver - 3")
#    plt.imshow(train_data1[0,0,:,:,i*10],cmap="bone")
# plt.figure()
# plt.imshow(train_data1[0,0,:,:,0],cmap="bone")
# plt.figure()
# plt.title("Kidney - 8")
# plt.imshow(train_data2[0,:,:,0],cmap="bone")
#
# plt.show()
# print "Finished"
