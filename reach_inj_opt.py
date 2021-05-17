### Injection optimization for the REACH module on SCExAO
### Author : Sebastien Vievard




import pygame, sys
from pygame.locals import *
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from astropy.io import fits
import struct 
import os
from PIL import Image
import time
import math as m
import copy
import datetime
from astropy.io import fits as pf
import subprocess
import os,sys
# from xaosim.scexao_shmlib import shm
from pyMilk.interfacing.isio_shmlib import SHM as shm
home = os.getenv('HOME')
sys.path.append(home+'/src/lib/python/')
sys.path.append(home+'/Documents/sebviev/Bench_tests/')
import conex_tt as conex_tt
import conex3 as conex
con = conex.conex()
from tqdm import tqdm
import imageio
from scipy import optimize
from scipy.optimize import curve_fit


plt.ion()

class reach_inj(object):
    
    def __init__(self):###############################################################################################
        
        self.flux=shm("ird_flux_val")

        ## Conex for Theta
        self.conex_id_theta="/dev/serial/by-id/usb-Newport_CONEX-AGP_A64MV6OB-if00-port0"
        ## Connex for Phi
        self.conex_id_phi="/dev/serial/by-id/usb-Newport_CONEX-AGP_A6WMSQ3G-if00-port0"
        ## Conex for pcfi_len --> focus of the fiber
        self.conexidz="/dev/serial/by-id/usb-Newport_CONEX-AGP_A606QDT0-if00-port0u"
        ## Tip/Tilt mount adress
        self.adress_TT = "http://133.40.163.196:50002"
        ## NOTE : For TT, max values +/- 0.75
        ## NOTE : move(x,y) --> X = theta, Y = Phi

    def save_info(self,info, save_name):######################################################################################
        import json
        
        # Make it work for Python 2+3 and with Unicode
        import io
        try:
            to_unicode = unicode
        except NameError:
            to_unicode = str

        # Write JSON file
        with io.open(save_name, 'w', encoding='utf8') as outfile:
            str_ = json.dumps(info,
                              indent=4, sort_keys=True,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))
                
    def spiral(self,a):##############################################################################################
        sp = np.zeros((a**2,2))
        switch = np.zeros(a**2)
        x = y = 0
        dx = 0
        dy = -1
        for i in range(a**2):
            sp[i,:] = np.array([x,y], dtype=float)/(a-1)*2
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
                switch[i] = True
                dx, dy = -dy, dx
            else:
                switch[i] = False
            x, y = x+dx, y+dy
        return sp,switch

    def show_optim(self,res_opt,ird_channel_opt,Target):######################################################################################
        plt.figure()
        plt.subplot(4,2,1)
        idplt=1
        cmin=np.min(self.res_optim)
        cmax=np.max(self.res_optim[:,:,ird_channel_opt])
        for id in range(8):
           plt.subplot(4,2,idplt)
           plt.imshow(self.res_optim[:,:,id],origin='lower')
           plt.ylabel('#'+str(id+1))
           # plt.clim([cmin,cmax])
           idplt+=1


        
        self.date=datetime.datetime.today().strftime('%Y-%m-%d')
        self.clock=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path_save='/home/scexao/Documents/sebviev/REACH_injection/data/Optim_maps/'+self.date+'/'+self.clock+'_'+Target+'/'
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        #plt.savefig(path_save+'optim_result.png',orientation='landscape')
        for i in range(8):
        #    plt.imsave(path_save+'optim_'+self.clock+'_'+str(i+1)+'.pdf', self.res_optim[:,:,i])
            hdu=fits.PrimaryHDU(self.res_optim[:,:,i])
            hdu.writeto(path_save+'optim_'+self.clock+'_'+str(i+1)+'.fits')

    def optimization_slow(self,phi0=0., theta0=0., window_mas=0.05, npt=5, n_raw=1, ird_channel_opt=0, Target='Name_of_Target'):######################
        #############################
        #      Unit conversion      #
        #############################

        step_mas 			= window_mas/(npt-1)
        sp, switch          = self.spiral(npt)
        sp                  = sp*(window_mas/2.)
        
        xi,yi               = 0,0
        self.res_optim_raw  = np.zeros((npt, npt, n_raw, 8))
        self.res_optim      = np.zeros((npt, npt, 8))
        indphi0             = int(npt//2.)
        indtheta0           = int(npt//2.)

        self.map_phi        = np.zeros((npt,npt))
        self.map_theta      = np.zeros((npt,npt))
        
        #step2mas            = 5300
        #mas2step            = 1/5300
        step2mas            = 1
        mas2step            = 1
        

        #############################
        #      First position       #
        #############################

        # con.open(self.conex_id_phi)
        # con.move(phi0, "", False)
        conex_tt.move(theta0,phi0,address=self.adress_TT)
        time.sleep(0.2)
        # con.close()

        # con.open(self.conex_id_theta)
        # con.move(theta0, "", False)
        # time.sleep(0.1)
        # con.close()

        cnt  = 0
        cnt1 = 0

        #print('Position: x:',phi0,'y:',theta0)
        
        self.map_phi[indphi0,indtheta0] = phi0
        self.map_theta[indphi0,indtheta0] = theta0

        for r in range(n_raw):
            while (cnt1 <= cnt):
                cnt1 = self.flux.get_counter()
            cnt = cnt1
            self.res_optim_raw[indphi0,indtheta0,r,:]   = (self.flux.get_data(True, True, timeout = 1.))#[:,0]
            #time.sleep(0.002)
            
           
        self.res_optim[indphi0,indtheta0,:] = np.mean(self.res_optim_raw[indphi0,indtheta0,:,:],axis=0)
        #print(self.res_optim[int(indphi0)-1,int(indtheta0)-1])
        #print('Index: x:',np.ceil(indphi0-1),'y:',np.ceil(indtheta0-1))
        #############################
        #  Start spiral movement    #
        #############################
        for i in tqdm(range(1,npt**2)):
        # for i in range(1,npt**2):

            xi                                        = phi0+sp[i,0]*mas2step
            yi                                        = theta0+sp[i,1]*mas2step
            #print('Position: x:',xi,'y:',yi)
            indphi                                    = indphi0+int(sp[i,0]/(window_mas/2.)*(npt//2.))
            indtheta                                  = indtheta0+int(sp[i,1]/(window_mas/2.)*(npt//2.))
            self.map_phi[indphi,indtheta] = xi
            self.map_theta[indphi,indtheta] = yi
            #print('Index: x:',np.ceil(indphi-1),'y:',np.ceil(indtheta-1))
            if i==1:
                # con.open(self.conex_id_phi)
                axis = "x"
                # con.move(xi, "", False)
                conex_tt.move(yi,xi,address=self.adress_TT)
                #print(i, axis, xi-phi0, yi-theta0)
            if i > 1:
                if switch[i-1]:
                    # con.close()
                    if axis == "y":
                        # con.open(self.conex_id_phi)
                        axis = "x"
                    else:
                        # con.open(self.conex_id_theta)
                        axis = "y"
                if axis == "x":
                    # con.move(xi, "", False)
                    conex_tt.move(yi,xi,address=self.adress_TT)
                else:
                	conex_tt.move(yi,xi,address=self.adress_TT)
                    # con.move(yi, "", False)

            
            time.sleep(0.2)
            #############################
            #     Take Flux values      #
            ############################# 
            cnt  = 0
            cnt1 = 0
            for r in range(n_raw):
                while (cnt1 <= cnt):
                    cnt1 = self.flux.get_counter()
                cnt = cnt1
                self.res_optim_raw[indphi,indtheta,r,:]  = (self.flux.get_data(True, True, timeout = 1.))#[:,0]
                #time.sleep(0.002)
            
           
            self.res_optim[indphi,indtheta,:]            = np.mean(self.res_optim_raw[indphi,indtheta,:,:],axis=0)
            #print(self.res_optim[int(indphi)-1,int(indtheta)-1])
            
        #############################
        #  Show/save Optimization   #
        #############################
        
        self.show_optim(self.res_optim,ird_channel_opt,Target)


        #############################
        #    Extract optimal pos    #
        #############################

        indphi_opt, indtheta_opt  = self.process_optim_map(self.res_optim[:,:,ird_channel_opt], Target, window_mas, np.around(step_mas,decimals=2))
        phi_opt 	= phi0  -(indphi0  -indphi_opt)  *step_mas
        theta_opt 	= theta0-(indtheta0-indtheta_opt)*step_mas
        print('Optimal process position in phi:',phi_opt, 'in theta:',theta_opt)



        if phi_opt < np.around(np.min(self.map_phi),decimals=4) or phi_opt > np.around(np.max(self.map_phi),decimals=4):
	        indphi_opt, indtheta_opt = np.unravel_index(self.res_optim[:,:,ird_channel_opt].argmax(),(npt,npt))
        	phi_opt = self.map_phi[indphi_opt, indtheta_opt]
        	print('Warning : Phi optimal value out of the box')

        if theta_opt < np.around(np.min(self.map_theta),decimals=4) or theta_opt > np.around(np.max(self.map_theta),decimals=4):
	        indphi_opt, indtheta_opt = np.unravel_index(self.res_optim[:,:,ird_channel_opt].argmax(),(npt,npt))
	        theta_opt = self.map_theta[indphi_opt, indtheta_opt]
	        print('Warning : Theta optimal value out of the box')

        #scan_mas=(((np.max(self.map_phi)-np.min(self.map_phi))*1e3)/10.)*53
        #step_mas=(((self.map_phi[1,0]-self.map_phi[0,0])*1e3)/10.)*53
        scan_mas = 1
        step_mas = 1
        print('Scan in x from ',np.around(np.min(self.map_phi),decimals=4),'to',np.around(np.max(self.map_phi),decimals=4), '--', np.around(scan_mas,decimals=0),'mas')
        print('Scan in y from ',np.around(np.min(self.map_theta),decimals=4),'to',np.around(np.max(self.map_theta),decimals=4))
        print('Step:',np.around((self.map_phi[1,0]-self.map_phi[0,0])*1e3,decimals=1),'um', '--', np.around(step_mas,decimals=0),'mas' )
        
        print('Optimal position in phi:',np.around(phi_opt, decimals = 6), 'in theta:',np.around(theta_opt, decimals=6))
        
        
        #############################
        #    Move to optimal pos    #
        #############################
        conex_tt.move(theta_opt,phi_opt,address=self.adress_TT)
        # con.open(self.conex_id_phi)
        # con.move(phi_opt, "", False)
	    
        # con.close()
        # con.open(self.conex_id_theta)
        # con.move(theta_opt, "", False)

        # con.close()


        #############################
        #      Save info optim      #
        #############################
        info={'Target'                : Target,
              'phi init'              : phi0,
              'theta init'            : theta0,
              'Windows size (mas)'    : window_mas,
              'Step size (mas)'       : step_mas,
              'Number frames'         : n_raw,
              'phi optimal'           : phi_opt,
              'theta optimal'         : theta_opt,
              }
        
        #date=datetime.datetime.today().strftime('%Y-%m-%d')
        #clock=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.save_info(info, '/home/scexao/Documents/sebviev/REACH_injection/data/Optim_maps/'+self.date+'/'+self.clock+'_'+Target+'/info_'+self.clock+'_'+Target+'.txt')
        return(theta_opt,phi_opt) #SEB, I added this line for my code -Julien
       
    def process_optim_map(self, data, Target, window_mas, step_mas):######################################################################################


        def gaussian(height, center_x, center_y, width_x, width_y):
            """Returns a gaussian function with the given parameters"""
            width_x = float(width_x)
            width_y = float(width_y)
            return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
            
        def moments(data):
            """Returns (height, x, y, width_x, width_y)
            the gaussian parameters of a 2D distribution by calculating its
            moments """
            total = data.sum()
            X, Y = np.indices(data.shape)
            x = (X*data).sum()/total
            y = (Y*data).sum()/total
            col = data[:, int(y)]
            width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
            row = data[int(x), :]
            width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
            height = data.max()
            return height, x, y, width_x, width_y

        def fitgaussian(data):
            """Returns (height, x, y, width_x, width_y)
            the gaussian parameters of a 2D distribution found by a fit"""
            params = moments(data)
            errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                               data)
            p, success = optimize.leastsq(errorfunction, params)
            return p
            

    
        image=data
        plt.figure()
        plt.imshow(image,origin='lower')
        params=fitgaussian(image)
        fit=gaussian(*params)
        
        plt.contour(fit(*np.indices(image.shape)), cmap=plt.cm.copper)
        ax=plt.gca()
        (height,x,y,width_x,width_y)=params
        
        #print('(%.1f,%.1f)'%(x,y))
        plt.gca().set_title('Optimization map - '+Target)#+'- Scan='+str(window_mas)+'mas Step='+str(step_mas)+'mas')
        plt.text(0.95,0.05,"""
        Center:(%.1f,%.1f)
        Width_x:%.1f
        Width_y:%.1f"""%(x,y,width_x,width_y),fontsize=10,horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes,color='w')
        
        
        path_save='/home/scexao/Documents/sebviev/REACH_injection/data/Optim_maps/'+self.date+'/'+self.clock+'_'+Target+'/'
        plt.savefig(path_save+Target+'_processed.png')

        return x,y




##### Main of test #####

#### First you initialize the object by giving it a name you will use. Here I am going to call it "ird"
r_inj=reach_inj()


