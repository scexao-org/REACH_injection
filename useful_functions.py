from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import affine_transform




##### IMAGE OPERATIONS


def Fourier_resample(input_data,new_shape):
	# Resample/rescale a 2D image thanks to the fourier resample technique

	# --------------------------------------------------------------------------
	# Extraction of shape of input data + creation useful grids for processing
	# --------------------------------------------------------------------------

	len_x, len_y = input_data.shape
	grid_from    = make_pupil_grid((len_x,len_y))
	grid_to      = make_pupil_grid((new_shape[0],new_shape[1]))

	# --------------------------------------------------------------------------
	# Reshaping
	# --------------------------------------------------------------------------

	fft = FastFourierTransform(grid_from,q=1,fov=1)
	mft=MatrixFourierTransform(grid_to,fft.output_grid)
	reshaped_data=mft.backward(fft.forward(Field(input_data.ravel(),grid_from)))

	return np.reshape(reshaped_data.real,[int(new_shape[0]),int(new_shape[1])])


def eclat_image(data):

	sszz = data.shape
	
	if len(sszz) is 2:
		nl = sszz[0]
		nc = sszz[1]
		#gami = np.roll(data,[int(nl/2),int(nc/2)])
		gami = np.roll(data,int(nc/2-1),axis=0)
		gami = np.roll(gami,int(nl/2-1),axis=1)

	if len(sszz) is 3:
		print('No cube handled yet')
		nl = sszz[1]
		nc = sszz[2]

	return gami


def cent_rot(im, rot, rotation_center):
    '''
    cen_rot - takes a cube of images im, and a set of rotation angles in rot,
    and translates the middle of the frame with a size dim_out to the middle of
    a new output frame with an additional rotation of rot.
    '''
    # converting rotation to radians
    a = np.radians(rot)

    # make a rotation matrix
    transform = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])[:,:]
    # calculate total offset for image output

    c_in = rotation_center#center of rotation
    # c_out has to be pre-rotated to make offset correct

    offset = np.dot(transform, -c_in) + c_in
    offset = (offset[0], offset[1],)

    # perform the transformation
    dst = affine_transform(im, transform, offset=offset)

    return dst

##### UNIT CONVERSION

def rad2microns(array,wavelength):
    # Converts radians into microns, depending on the selected wavelength 
    #
    # Wavelength should be in microns

    phase_microns=(array*wavelength)/(2*np.pi)

    return phase_microns


def reshape_to_real_dmmap(dmmap):

	##### Info pupil
	rot_angle             	= 6.25                                  # In degrees
	size_pup_dm           	= 44
	size_dm               	= 50
	dmmap_command			= np.zeros([size_dm,size_dm]) 
	mask_pupil_dm           = np.zeros([size_dm,size_dm]) 


	##### Get real pupil mask
	real_pupil_dm         	= fits.getdata('/media/data/Sebviev/LWE/data/Modes/pupil_on_dm.fits')
	size_pup 				= real_pupil_dm.shape[1]

	##### Rotate real pupil mask
	real_pupil_dm_rotated 	= cent_rot(real_pupil_dm,rot_angle,np.array([size_pup/2.,size_pup/2.]))

	##### Resample to actual size of the pupil
	real_pupil_resize          	= Fourier_resample(real_pupil_dm_rotated,[size_pup_dm,size_pup_dm])
	real_pupil_resize[np.where(real_pupil_resize < 0.5)] = 0.
	real_pupil_resize[np.where(real_pupil_resize > 0.5)] = 1.

	##### Immerge the real pupil into the DM map
	mask_pupil_dm[1:size_pup_dm+1,2:size_pup_dm+2] = real_pupil_resize

	##### Multiply given dmmap by the mask

	if np.size(np.shape(dmmap)) == 2:
		dmmap_current 									= dmmap
		##### Reshape / rotate current map to pupil map on the DM
		size_map                						= np.shape(dmmap_current)[1]
		dmmap_rotated     								= cent_rot(dmmap_current,rot_angle,np.array([size_map/2.,size_map/2.]))
		dmmap_radians     								= Fourier_resample(dmmap_rotated,[size_pup_dm,size_pup_dm])
		dmmap_command[1:size_pup_dm+1,2:size_pup_dm+2] 	= dmmap_radians
		pupil_map_on_dm 								= mask_pupil_dm*dmmap_command

	if np.size(np.shape(dmmap)) > 2:
		size_cube 		= np.shape(dmmap)
		print(size_cube[0],size_cube[1],size_cube[2])
		pupil_map_on_dm = np.zeros([size_cube[0],size_dm,size_dm])
		for i in range(size_cube[0]):
			dmmap_current = dmmap[i,:,:]
			##### Reshape / rotate current map to pupil map on the DM
			size_map                						= np.shape(dmmap_current)[1]
			dmmap_rotated     								= cent_rot(dmmap_current,rot_angle,np.array([size_map/2.,size_map/2.]))
			dmmap_radians     								= Fourier_resample(dmmap_rotated,[size_pup_dm,size_pup_dm])
			dmmap_command[1:size_pup_dm+1,2:size_pup_dm+2] 	= dmmap_radians
			pupil_map_on_dm[i,:,:] 							= mask_pupil_dm*dmmap_command



	return pupil_map_on_dm



