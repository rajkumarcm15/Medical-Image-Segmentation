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
from numpy import mean,std,tile,random
from medpy.io import load
import numpy as np
import time
import threading
import logging

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
        self.file_names_main = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        self.TEST_INDEX = 100
        self.file_names = []
        if training:
            self.file_names = self.file_names_main[:self.TEST_INDEX]
            # print self.file_names
        else:
            self.file_names = self.file_names_main[self.TEST_INDEX:]

        n_data = len(self.file_names)
        self.indices = np.random.permutation(n_data)
        self.p_shape = p_shape
        self.augment_sigma = augment_sigma
        self.numeric_targ = numeric_targ
        self.log = logging.getLogger()
        # self.get_next()
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
        # thread = threading.Thread(target=self.process_data,args=(self.id,self.p_shape))
        thread = threading.Thread(target=self.process_data(self.id,self.p_shape))
        thread.daemon = True
        thread.start()

    def shift_pointer(self,N_Z,b_pointer,dim_pointer):

        if b_pointer == len(self.indices)-1 and dim_pointer == N_Z-1:
            b_pointer = 0
            dim_pointer = 0
            self.complete = True
        elif dim_pointer == N_Z-1:
            b_pointer += 1
            dim_pointer = 0
        else:
            dim_pointer += 1

        return b_pointer, dim_pointer

    def get_pointer(self):

        return self.b_pointer, self.dim_pointer

    def update_pointer(self,b_pointer,dim_pointer):

        self.b_pointer = b_pointer
        self.dim_pointer = dim_pointer

        return True

    def refine_pos(self,pos):
        x_rad = self.p_shape[0]/2
        y_rad = self.p_shape[1]/2
        block = zip(pos[0],pos[1])
        positions = []
        max_x = 512-x_rad
        max_y = 512-y_rad
        min_x = 0+x_rad
        min_y = 0+y_rad
        # if self.id != 0:
        positions = filter(lambda x: (x[0] <= max_x and x[0] >= min_x and x[1] <= max_y and x[1] >=min_y ), block)
        # else:

            # positions = filter(lambda x: (x[0] <= max_x and x[0] >= min_x and x[1] <= max_y and x[1] >=min_y ), block)

        return positions

    def process_data(self,id,p_shape):
        nx = None
        ny = None
        N_Z = 371
        imgs = []
        lbls = []
        # s_c -> samples computed
        s_c = 0
        offsets = []
        while True:
            b_pointer_old,dim_pointer = self.get_pointer()
            b_pointer_new, dim_pointer = self.shift_pointer(N_Z,b_pointer_old,dim_pointer)
            b_pointer = b_pointer_new
            # if ( b_pointer_old == b_pointer_new ) and self.temp_img != None:
            #     b_pointer = b_pointer_new
            # else:
            if ( b_pointer_old != b_pointer_new ) or ( self.temp_img is None ):
                b_pointer = b_pointer_new
                X_fn = self.dir_path+"/aff_imgs/"+self.file_names[self.indices[b_pointer_new]]
                Y_fn = self.dir_path+"/aff_lbls/"+self.file_names[self.indices[b_pointer_new]]
                self.temp_img = self.load_data(X_fn)
                self.temp_lbl = self.load_data(Y_fn)

            img = self.temp_img
            lbl = self.temp_lbl
            nx = img.shape[0]
            ny = img.shape[1]

            d_p = dim_pointer

            for z in range(d_p,N_Z):
                # print("file: %d, dim: %d"%(b_pointer,z))
                img_ = img[:,:,z]
                lbl_ = lbl[:,:,z]
                pos = np.where(lbl_ == id)
                positions = self.refine_pos(pos)

                if len(positions) >= 10:
                    # if self.id == 3:
                    #     print "file_name: ",self.file_names[self.indices[b_pointer]]
                    #     print("dim: %d"%(z))
                    # print len(positions),", id: ",id

                    # positions =  [[x,y] for x,y in zip(pos[0],pos[1])]
                    if (s_c+len(positions)) <= self.n_samples:
                        s_c += len(positions)
                    else:
                        np.random.shuffle(positions)
                        positions = positions[:(self.n_samples-s_c)]
                        s_c = self.n_samples

                    offsets.append(positions)
                    imgs.append(img_)

                # else:
                #     print "else: ",len(pos[0])
                if s_c == self.n_samples:
                    break

            # When the loop finishes update the dim_pointer
            # because the current point is updated by z
            updated = self.update_pointer(b_pointer,z)


            if s_c == self.n_samples:
                break

        # The while loop would be finished here
        imset = []
        for i in range(len(imgs)):
            tmp_patches = self.extract_patches(imgs[i],offsets[i],p_shape)
            tmp_patches = self.augment_data(tmp_patches)
            n_imgs = len(tmp_patches)
            imset.append(np.concatenate(tmp_patches))

        n_imgs = len(imset)
        imset = np.concatenate(imset,axis=0)


        temp_lbl = [0,0,0]
        if id == 3:
            temp_lbl[0] = 1
        elif id == 8:
            temp_lbl[1] = 1
        else:
            temp_lbl[2] = 1

        lblset = np.tile(temp_lbl,[imset.shape[0]/p_shape[0],1])

        self.train_data = imset
        self.train_targ = lblset

        # print "Finished: ",id
        self.buffer_filled = True

    def load_data(self,file_name):
        epi_img,_ = load(file_name)
        return epi_img

    def extract_patches(self,img,offsets,p_shape):
        im_set = []
        p_radius = [x/2 for x in p_shape]
        nx = img.shape[0]
        ny = img.shape[1]
        for i in range(len(offsets)):
            _os = offsets[i]
            x_min = _os[0] - p_radius[0]
            x_max = _os[0] + p_radius[0]
            y_min = _os[1] - p_radius[1]
            y_max = _os[1] + p_radius[1]

            im_set.append(img[x_min:x_max,y_min:y_max])
        return im_set

    def augment_data(self,imgs):
        im_set = []
        for i in range(len(imgs)):
            temp_im = imgs[i].astype('float64')
            temp_im += np.random.normal(loc=0,scale=self.augment_sigma,size=temp_im.shape)
            # Vertical Flipping
            flipp_v = []
            for i in reversed(temp_im):
                flipp_v.append(i)
            flipp_v = np.matrix(flipp_v)

            # Horizontal Flipping
            flipp_h = []
            for i in reversed(temp_im.T):
                flipp_h.append(i)
            flipp_h = np.matrix(flipp_h).T

            im_set.append(temp_im)
            im_set.append(flipp_v)
            im_set.append(flipp_h)
        return im_set




####
#
dir = '/Users/Rajkumar/Documents/ISO/CNN/Project/Data'
_data1 = Data(3,dir,[112,112],numeric_targ=True,n_samples=1,training=True)
# # _data2 = Data(8,dir,[112,112],numeric_targ=True,n_samples=10)
while True:
   train_data1, train_targ1, c = _data1.get_next()
# # train_data2, train_targ2 = _data2.get_next()
print "Finished"