import pygame, sys
from pygame.locals import *
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import struct 
import os
from PIL import Image
import time
import math as m
import copy
import datetime as dt
from astropy.io import fits as pf
import subprocess
import os,sys
# from xaosim.scexao_shmlib import shm
from pyMilk.interfacing.isio_shmlib import SHM as shm

plt.ion()

class reach_pcfi(object):

    def __init__(self):
        print('Initializing camera shared memory...')
        self.cam = shm("ircam0")
        self.dark = shm("ircam0_dark")
        if not os.path.isfile("ird_flux_val"):
            os.system("creashmim %s %d %d" % ("ird_flux_val",1,8))
        try: 
            self.flux=shm("ird_flux_val",((8),np.float32),location=-1,shared=1)
        except:
            self.flux=shm('ird_flux_val',((8),np.float64),location=-1,shared=1)
        #self.flux = shm("/tmp/ird_flux_val.im.shm", data=np.zeros((1,8)),verbose=False)
        image=self.acqu_im()
        self.xs=image.shape[0]
        self.ys=image.shape[1]
        self.IRD_spot_position(image)
        self.flux_per_spot=np.zeros((8))
        print('end init')

    def Close_shm(self):
        #### Closes the shared memory 
        self.cam.close()
        self.dark.close()
        self.flux.close()

    def acqu_im(self,crop=None,show=None):
        #### Acquires an image with the Dark from the shared memory
        # Image acquisition
        image = self.cam.get_data(False, True, timeout = 1.).astype(float)
        # Dark subtraction (optional)
        im_dark = self.dark.get_data(False, True, timeout = 1.)
        image -= im_dark
        if show is True:
            plt.imsave("ircam.png", image)
            plt.figure(0)
            plt.imshow(image,aspect='auto')
        if crop is True:
            image_crop=np.zeros([4*int(self.irdc),self.ys])
            image_crop[:,:]=image[int(self.xc)-int(self.pird)+1:int(self.xc)+int(self.pird)-1,:]
            image=image_crop
        return image    
        


    def circle_mask(self,x0,y0,R,support):
        #### Creates a mask on a support with (x0,y0) coordinates as center, and 
        #### a radius R
        x_size=support.shape[0]
        y_size=support.shape[1]
        mask=np.zeros([x_size,y_size])
        for i in range(x_size):
            for j in range(y_size):
                temp=(i-x0)**2.+(j-y0)**2.
                if temp<R**2.:
                    mask[i,j]=1.
        
        return mask

    def IRD_spot_position(self,image):
        #### Defines the spot position on chuckam, creates masks for each spot
        #IRD parameters
        z1=1.
        dird = 64.*z1
        xird = 7.5*z1 # before : -3.5
        yird = -11.5*z1
        self.pird = dird/7.
        self.irdc = self.pird/2.
        self.ircam_IRD_spots=np.zeros([self.xs,self.ys,8])
        self.ircam_IRD_spots_crop=np.zeros([4*int(self.irdc),self.ys,8])
        for i in range(8):
            self.yc=int(self.ys/2+xird+(i-3.5)*self.pird)
            self.xc=int(self.xs/2+yird)
            self.ircam_IRD_spots[:,:,i]=self.circle_mask(self.xc,self.yc,self.irdc,image)
            self.ircam_IRD_spots_crop[:,:,i]=self.ircam_IRD_spots[int(self.xc)-int(self.pird)+1:int(self.xc)+int(self.pird)-1,:,i]
        #plt.figure(0)
        #plt.imshow(np.sum(self.ircam_IRD_spots_crop, axis=2))


    def Flux_values(self,image):
        #### Extracts the total flux in each spot area
        
        for i in range(8):
            self.flux_per_spot[i]=(image*self.ircam_IRD_spots_crop[:,:,i]).sum()
        self.flux.set_data(self.flux_per_spot.astype(np.float32))




####### real-time code

reach=reach_pcfi()
cnt = 0
cnt1 = 0
while True:
    while (cnt1 <= cnt):
        cnt1 = reach.cam.get_counter()
    cnt = cnt1
    image=reach.acqu_im(crop=True)
    flux=reach.Flux_values(image)
