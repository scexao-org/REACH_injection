### Optimization of the Zernikes applied on SCExAO deformable mirror for the REACH injection
### Author : Sebastien Vievard



import pygame, sys
from pygame.locals import *
import numpy as np
import matplotlib
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
home = os.getenv('HOME')
sys.path.append(home+'/src/lib/python/')
import conex3 as conex
con = conex.conex()
from tqdm import tqdm
import imageio
from scipy import optimize
from scipy.optimize import curve_fit
from lmfit import Model
from lmfit import Parameters
from lmfit.models import GaussianModel

from pyMilk.interfacing.isio_shmlib import SHM as shm
import bench_functions as bf


# -----------------------------------------------------------------------------
# Initialize shared memory 
# -----------------------------------------------------------------------------
# Flux from REACH photometry
flux 	= shm("ird_flux_val")
# DM, channel 4
# dm 		= shm("dm00disp04")

# -----------------------------------------------------------------------------
# Values of interest for the phase map
# -----------------------------------------------------------------------------
RMS_val = 0.5     # RMS value of the perturbation, in radians  // USELESS HERE
N_modes = 23      # Number of modes desired for the phase map (Defoc start at #12 )

# -----------------------------------------------------------------------------
# Select Zernike
# -----------------------------------------------------------------------------
# Here is to choose what kind of perturbatin we want to introduce: 
# 'LWE'      - Low wind effect modes - N_modes should be at least 12
# '#Zernike' - only one Zernike - N_modes should be at least 12+(number of Zernike-3)
#                                 ex: defoc(Z4) : N_modes at least 12+(4-3)=13
# 'NCPA'      - Only NCPA - the 12 LWE modes will be put to 0 
# -----------------------------------------------------------------------------
#Type_aberration='LWE'
#Type_aberration='Z10'
Type_aberration='NCPA'


# -----------------------------------------------------------------------------
# Values of interest for the test
# -----------------------------------------------------------------------------
step 				= 0.1
ai 					= np.arange(-1,1+step,step)
n_moy				= 20
flux_test 			= np.zeros([N_modes, np.size(ai),n_moy])
flux_curve_fit 		= np.zeros([N_modes, np.size(ai)])
optim_zern_coeff 	= np.zeros(N_modes)

phi					= 0.
theta 				= 0.

# Load the Zernike modes
Modes = fits.getdata('/home/scexao/Documents/sebviev/ird_pcfi/dm_subaru_modes_resize.fits')

# -----------------------------------------------------------------------------
# Loop on the Zernikes
# -----------------------------------------------------------------------------
for z in range(N_modes-12):
	print('Mode Z'+str(z+4))
	for a in range(np.size(ai)):
		# print('ai ='+str(ai[a]))
		coeff_zernike = np.zeros(12+N_modes)
		coeff_zernike[12+z] = ai[a]
		# print(coeff_zernike)

		# -----------------------------------------------------------------------------
		# Making the phase map to send to the DM
		# -----------------------------------------------------------------------------
		DM_command=bf.make_phase_map(N_modes,ai[a],coeff_zernike,Type_aberration,Modes= Modes)

		# -----------------------------------------------------------------------------
		# Push the phase to the dm
		# -----------------------------------------------------------------------------
		bf.apply_dm_phasemap(DM_command, 0.5, wavelength_microns = 1.5)
		time.sleep(0.1)

		# -----------------------------------------------------------------------------
		# Get the flux values from REACH photometry
		# -----------------------------------------------------------------------------
		for i in range(n_moy):
			flux_test[z,a,i] = flux.get_data(True, True, timeout = 1.)[0]

# Zero the DM here once the loop is done
bf.apply_dm_phasemap(DM_command*0., 0.5, wavelength_microns = 1.5)

# -----------------------------------------------------------------------------
# Results fitting
# -----------------------------------------------------------------------------
mean_flux = np.mean(flux_test,axis=2)

def gaussian(x,amp,cen,wid):
	return ((amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-cen)**2 / (2*wid**2)))
gauss_model = Model(gaussian)

# Fit the data points with a Gaussian
for z in range(N_modes-12):
	fit_temp = gauss_model.fit(mean_flux[z,:], x=np.arange(0,np.size(ai)), 
								amp = 1, cen = np.size(ai)/2, wid = np.size(ai)/2 )
	flux_curve_fit[z,:] = fit_temp.best_fit
	optim_zern_coeff[z+12] = fit_temp.params['cen'].value

# print(optim_zern_coeff)

optim_zern_coeff[12:] = (-1)+step*optim_zern_coeff[12:]
print(np.around( (optim_zern_coeff/6.28)*1500,decimals=2))

# -----------------------------------------------------------------------------
# Making the optimized phase map to send to the DM
# -----------------------------------------------------------------------------
DM_command=bf.make_phase_map(N_modes,ai[a],optim_zern_coeff,Type_aberration, Modes= Modes,display=True)

# -----------------------------------------------------------------------------
# Push the phase to the dm
# -----------------------------------------------------------------------------
# bf.apply_dm_phasemap(DM_command, 0.5, wavelength_microns = 1.5)


# -----------------------------------------------------------------------------
# Save stuff
# -----------------------------------------------------------------------------


date=datetime.datetime.today().strftime('%Y-%m-%d')
clock=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
path_save='/home/scexao/Documents/sebviev/ird_pcfi/data/Optim_Zernikes/'+date+'/'+clock+'_Phi_'+str(phi)+'_Theta_'+str(theta)+'/'
if not os.path.exists(path_save):
	os.makedirs(path_save)


plt.rcParams.update({'font.size':8})
# plt.figure(figsize=(13,10))
plt.figure()
for z in range(N_modes-12):
	plt.subplot(np.int(np.sqrt(N_modes-12))+1, np.int(np.sqrt(N_modes-12))+1, z+1)
	plt.plot(ai,flux_curve_fit[z,:])
	plt.plot(ai,mean_flux[z,:], 'r+')
	#plt.ylim(1e4,6e4)
	# plt.text(-1,4e4, 'Mode Z'+str(z+4))
	plt.ylabel('Flux')
	plt.xlabel('Zernike coeff. (rad) Mode Z'+str(z+4))
	plt.savefig(path_save+'Zernike_optim_plot.png')
plt.tight_layout()

np.save(path_save+'Zernike_values.npy', optim_zern_coeff)


# -----------------------------------------------------------------------------
# Close shared memory
# -----------------------------------------------------------------------------

# dm.close()


