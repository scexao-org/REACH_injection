import os
import numpy as np
import matplotlib.pyplot as plt 
import time
import datetime
from scipy import ndimage
from astropy.io import fits
# from xaosim.scexao_shmlib import shm
from pyMilk.interfacing.isio_shmlib import SHM as shm
import image_operation_functions as iof
from lmfit import Model

plt.ion()


############### IMAGE ACQUISITION

def acqu_chuck_image(N_images=1, display=False, save=False, save_name = 'Test'):
    # -----------------------------------------------------------------------------
    # Init camera and Dark shared memory
    # -----------------------------------------------------------------------------
    cam = shm("ircam0", verbose=False)
    dark = shm("ircam0_dark", verbose=False)
    #badpix =shm("/tmp/ircam0_badpixmap.im.shm", verbose=False)
    
    #cam = shm("ircam0")
    #dark = shm("ircam0_dark")
    #badpix =shm("ircam0_badpixmap")


    # Image acquisition
    image           = cam.get_data(True, True, timeout = 1.).astype(float)
    im_dark         = dark.get_data(True, True, timeout = 1.).astype(float)
    #bad_pix_image   = badpix.get_data(True, True, timeout = 1.).astype(float)
    
    #image = cam.get_data(True, True, timeout = 1.).astype(float)
    #im_dark = dark.get_data(True, True, timeout = 1.).astype(float)
    #bad_pix_image = badpix.get_data(True, True, timeout = 1.).astype(float)

    Im_tot = np.zeros([N_images, image.shape[0], image.shape[1]])

    for i in range(N_images):
        Im_tot[i,:,:] = cam.get_data(True, True, timeout = 1.).astype(float) - im_dark #- bad_pix_image
    
    Im_finale=np.mean(Im_tot,0)

    # -----------------------------------------------------------------------------
    # Display / Save images
    # -----------------------------------------------------------------------------
    if display is True:
        plt.figure(0)
        plt.imshow(Im_finale,aspect='auto')
    if save is True:
        date=datetime.datetime.today().strftime('%Y-%m-%d')
        clock=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_save='/mnt/sdata00/svievard/Data/Chuck_images/'+date+'/'
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
        path_save=folder_save+clock+'_'
        hdu = fits.PrimaryHDU(Im_finale)
        hdu.writeto(path_save+'im_chuck_'+save_name+'.fits',overwrite=True)
    
    # -----------------------------------------------------------------------------
    # Close camera and Dark shared memory
    # -----------------------------------------------------------------------------
    cam.close()
    dark.close()
    #badpix.close()

    return Im_finale

def acqu_vampires_images(N_images=1, Darks = False, correct_rotation = False, recenter = False, display=False, save=False, save_name = 'Test', debug=False):

    # -----------------------------------------------------------------------------
    # Init cameras shared memory
    # -----------------------------------------------------------------------------
    cam1        = shm("vcamim0") ## Defocused plane
    cam2        = shm("vcamim1") ## Focus plane
    
    # Image acquisition
    image1 = []
    image2 = []

    for i in range(N_images):
        image1.append(cam1.get_data(True, True, timeout = 1.).astype(float))
        image2.append(cam2.get_data(True, True, timeout = 1.).astype(float))

    image1 = np.array(image1)
    image2 = np.array(image2)    

    images_finales          = np.zeros([2, image1.shape[1], image1.shape[2]])
    images_finales[0,:,:]   = np.mean(image2,0) ### CAMERA 2 IS FOCAL IMAGE
    images_finales[1,:,:]   = np.mean(image1,0) ### CAMERA 1 IS DEFOCUSED IMAGE

    if Darks is True:
        cam1dark        = shm("vcamim0dark") 
        cam2dark        = shm("vcamim1dark") 

        images_finales[0,:,:] -= cam2dark.get_data(True, True, timeout = 1.).astype(float)
        images_finales[1,:,:] -= cam1dark.get_data(True, True, timeout = 1.).astype(float)

    # -----------------------------------------------------------------------------
    # Recenter images
    # -----------------------------------------------------------------------------
    if recenter is True :

        images_finales_1d_x     = np.sum(images_finales, axis=1)
        images_finales_1d_y     = np.sum(images_finales, axis=2)

        def gaussian(x, amp, cent, wid):
                return ((amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-cent)**2 /(2*wid**2)))

        gauss_model = Model(gaussian)

        ############ FITTINGS 
        fit_x_foc               = gauss_model.fit(images_finales_1d_x[0,:], 
                                                                                x =    np.arange(images_finales_1d_x[0,:].shape[0]),
                                                                                amp =  1, 
                                                                                cent = images_finales_1d_x[0,:].shape[0]/2,
                                                                                wid =  images_finales_1d_x[0,:].shape[0]/2)
        fit_curve               = fit_x_foc.best_fit
        cent_foc_x              = fit_x_foc.params['cent'].value
        if debug is True:
            plt.figure()
            plt.plot(images_finales_1d_x[0,:],'b')
            plt.plot(fit_curve,'r')

        fit_x_defoc     = gauss_model.fit(images_finales_1d_x[1,:], 
                                                                                x =    np.arange(images_finales_1d_x[1,:].shape[0]),
                                                                                amp =  1, 
                                                                                cent = images_finales_1d_x[1,:].shape[0]/2,
                                                                                wid =  images_finales_1d_x[1,:].shape[0]/2)
        fit_curve               = fit_x_defoc.best_fit
        cent_defoc_x            = fit_x_defoc.params['cent'].value
        if debug is True:
            plt.figure()
            plt.plot(images_finales_1d_x[1,:],'b')
            plt.plot(fit_curve,'r')


        fit_y_foc               = gauss_model.fit(images_finales_1d_y[0,:], 
                                                                                x =    np.arange(images_finales_1d_y[0,:].shape[0]),
                                                                                amp =  1, 
                                                                                cent = images_finales_1d_y[0,:].shape[0]/2,
                                                                                wid =  images_finales_1d_y[0,:].shape[0]/2)
        fit_curve               = fit_y_foc.best_fit
        cent_foc_y              = fit_y_foc.params['cent'].value
        if debug is True:
            plt.figure()
            plt.plot(images_finales_1d_y[0,:],'b')
            plt.plot(fit_curve,'r')

        fit_y_defoc     = gauss_model.fit(images_finales_1d_y[1,:], 
                                                                                x =    np.arange(images_finales_1d_y[0,:].shape[0]),
                                                                                amp =  1, 
                                                                                cent = images_finales_1d_y[1,:].shape[0]/2,
                                                                                wid =  images_finales_1d_y[1,:].shape[0]/2)
        fit_curve               = fit_y_defoc.best_fit
        cent_defoc_y            = fit_y_defoc.params['cent'].value
        if debug is True:
            plt.figure()
            plt.plot(images_finales_1d_y[1,:],'b')
            plt.plot(fit_curve,'r')
            print('Center image foc: '+str(cent_foc_x)+' ; '+str(cent_foc_y))
            print('Center image defoc: '+str(cent_defoc_x)+' ; '+str(cent_defoc_y))

        ############ APPLY SHIFTS
        shift_foc_x                     = cent_foc_x   - images_finales[0, :, :].shape[0]/2.
        shift_foc_y                     = cent_foc_y   - images_finales[0, :, :].shape[0]/2.
        shift_defoc_x                   = cent_defoc_x - images_finales[1, :, :].shape[0]/2.
        shift_defoc_y                   = cent_defoc_y - images_finales[1, :, :].shape[0]/2.
        
        images_finales[0, :, :] = ndimage.shift(images_finales[0, :, :], [-shift_foc_y   , -shift_foc_x  ] )
        images_finales[1, :, :] = ndimage.shift(images_finales[1, :, :], [-shift_defoc_y , -shift_defoc_x] )

    # -----------------------------------------------------------------------------
    # Correct Rotation
    # -----------------------------------------------------------------------------
    if correct_rotation is True:
        angle_cam1 = -0.55/np.pi*180
        angle_cam2 = -2.59/np.pi*180


        images_finales[0, :, :] = iof.cent_rot(images_finales[0, :, :],angle_cam1,np.array([images_finales[0, :, :].shape[0]/2,images_finales[0, :, :].shape[0]/2]))
        images_finales[0, :, :] = np.flipud(images_finales[0, :, :])
        images_finales[1, :, :] = iof.cent_rot(images_finales[1, :, :],angle_cam2,np.array([images_finales[0, :, :].shape[0]/2,images_finales[0, :, :].shape[0]/2]))

    # -----------------------------------------------------------------------------
    # Display / Save images
    # -----------------------------------------------------------------------------

    if display is True:
        plt.figure(0)
        plt.subplot(1,2,1)
        plt.imshow(images_finales[0,:,:])
        plt.subplot(1,2,2)
        plt.imshow(images_finales[1,:,:])
    if save is True:
        date=datetime.datetime.today().strftime('%Y-%m-%d')
        clock=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_save='/mnt/sdata00/svievard/Data/Vampires_images/'+date+'/'
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
        path_save=folder_save+clock+'_'
        hdu = fits.PrimaryHDU(images_finales)
        hdu.writeto(path_save+'im_vampires_'+save_name+'.fits',overwrite=True)

    # -----------------------------------------------------------------------------
    # Close camera and Dark shared memory
    # -----------------------------------------------------------------------------
    cam1.close()
    cam2.close()


    return images_finales

def recenter_vampires(images, display = False):

    # -----------------------------------------------------------------------------
    # Recenter images
    # -----------------------------------------------------------------------------

    images_finales_1d_x     = np.sum(images, axis=1)
    images_finales_1d_y     = np.sum(images, axis=2)

    def gaussian(x, amp, cent, wid):
            return ((amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-cent)**2 /(2*wid**2)))

    gauss_model = Model(gaussian)


    fit_x_foc               = gauss_model.fit(images_finales_1d_x[0,:], 
                                                                        x =    np.arange(images_finales_1d_x[0,:].shape[0]),
                                                                        amp =  np.max(images_finales_1d_x[0,:]), 
                                                                        cent = images_finales_1d_x[0,:].shape[0]/2,
                                                                        wid =  images_finales_1d_x[0,:].shape[0]/2)
    fit_curve               = fit_x_foc.best_fit
    cent_foc_x              = fit_x_foc.params['cent'].value
    if display is True:
        plt.figure()
        plt.plot(images_finales_1d_x[0,:],'b')
        plt.plot(fit_curve,'r')

    fit_x_defoc     = gauss_model.fit(images_finales_1d_x[1,:], 
                                                                x =    np.arange(images_finales_1d_x[1,:].shape[0]),
                                                                amp =  np.max(images_finales_1d_x[1,:]), 
                                                                cent = images_finales_1d_x[1,:].shape[0]/2,
                                                                wid =  images_finales_1d_x[1,:].shape[0]/2)
    fit_curve               = fit_x_defoc.best_fit
    cent_defoc_x            = fit_x_defoc.params['cent'].value
    if display is True:
        plt.figure()
        plt.plot(images_finales_1d_x[1,:],'b')
        plt.plot(fit_curve,'r')


    fit_y_foc               = gauss_model.fit(images_finales_1d_y[0,:], 
                                                                        x =    np.arange(images_finales_1d_y[0,:].shape[0]),
                                                                        amp =  np.max(images_finales_1d_x[0,:]), 
                                                                        cent = images_finales_1d_y[0,:].shape[0]/2,
                                                                        wid =  images_finales_1d_y[0,:].shape[0]/2)
    fit_curve               = fit_y_foc.best_fit
    cent_foc_y              = fit_y_foc.params['cent'].value
    if display is True:
        plt.figure()
        plt.plot(images_finales_1d_y[0,:],'b')
        plt.plot(fit_curve,'r')

    fit_y_defoc     = gauss_model.fit(images_finales_1d_y[1,:], 
                                                                x =    np.arange(images_finales_1d_y[0,:].shape[0]),
                                                                amp =  np.max(images_finales_1d_x[1,:]), 
                                                                cent = images_finales_1d_y[1,:].shape[0]/2,
                                                                wid =  images_finales_1d_y[1,:].shape[0]/2)
    fit_curve               = fit_y_defoc.best_fit
    cent_defoc_y            = fit_y_defoc.params['cent'].value
    if display is True:
        plt.figure()
        plt.plot(images_finales_1d_y[1,:],'b')
        plt.plot(fit_curve,'r')

    if display is True:
        print('Center image foc: '+str(cent_foc_x)+' ; '+str(cent_foc_y))
        print('Center image defoc: '+str(cent_defoc_x)+' ; '+str(cent_defoc_y))

    shift_foc_x                     = cent_foc_x   - images[0, :, :].shape[0]/2.
    shift_foc_y                     = cent_foc_y   - images[0, :, :].shape[0]/2.
    shift_defoc_x                   = cent_defoc_x - images[1, :, :].shape[0]/2.
    shift_defoc_y                   = cent_defoc_y - images[1, :, :].shape[0]/2.

    return shift_foc_x, shift_foc_y, shift_defoc_x, shift_defoc_y

def dark_vampires():
    # -----------------------------------------------------------------------------
    # Init cameras shared memory
    # -----------------------------------------------------------------------------


    im_darks = acqu_vampires_images(N_images=100)

    try:
        cam1        = shm("vcamim0dark") 
        cam2        = shm("vcamim1dark") 
    except:
        cam1        = shm("vcamim0dark", ((im_darks.shape[1], im_darks.shape[2]), np.float64), location =-1, shared=1 ) 
        cam2        = shm("vcamim1dark", ((im_darks.shape[1], im_darks.shape[2]), np.float64), location =-1, shared=1) 

    cam1.set_data(im_darks[1,:,:])
    cam2.set_data(im_darks[0,:,:])


############### GENERATE PHASE MAPS

def make_phase_map(N_modes,RMS,coeff_zernike,Type_aberration,Modes= False,display=False,save=False, save_name = 'Test'):
    # Generates a phase map with N_modes modes (including LWE modes) 
    # with a random distrib. of RMS rns

    # ----------------------------------------------------------------------------
    # Wavelength of the test
    # ----------------------------------------------------------------------------

    #wavelength_microns= 1.550                                  # H band
    wavelength_microns= 0.750
    rms_microns       = rad2microns(RMS,wavelength_microns)    # RMS value in microns
    rot_angle         = 6.25                                  # In degrees
    
    
    # -----------------------------------------------------------------------------
    # Initialize the image containing the phase map
    # -----------------------------------------------------------------------------
    if Modes is not False:
        Modes_subaru = Modes
    else:
        # Modes_subaru       = fits.getdata('/mnt/sdata00/svievard/Telescope_pupil/Subaru/dm_subaru_modes_resize.fits')
        Modes_subaru       = fits.getdata('/mnt/sdata00/svievard/Telescope_pupil/Subaru/dm_subaru_modes_resize.fits')
    size_pup           = Modes_subaru.shape[1]
    modes_zern         = np.zeros([N_modes,int(size_pup**2)])  # Zernike modes in the pupil
    
    # -----------------------------------------------------------------------------
    # Load the modes (number of modes = N_modes) selected for the desired phase map
    # -----------------------------------------------------------------------------
    for z in range(N_modes):
        modes_zern[z,:] = Modes_subaru[z,:,:].reshape(size_pup**2)

    # -----------------------------------------------------------------------------
    # Create the phase map from the modes + RMS value input
    # -----------------------------------------------------------------------------
    #coeff_zernike=RMS*np.random.randn(N_modes)
    global_piston = (coeff_zernike[0]+coeff_zernike[3]+coeff_zernike[6]+coeff_zernike[9])/4.
    coeff_zernike[0] -= global_piston
    coeff_zernike[3] -= global_piston
    coeff_zernike[6] -= global_piston
    coeff_zernike[9] -= global_piston
    phase_array=np.zeros([N_modes,int(size_pup**2)])
    
    for z in range(N_modes):
        phase_array[z,:]=np.dot(modes_zern[z,:],coeff_zernike[z])


    # -----------------------------------------------------------------------------
    # Filter modes if some of them are not wanted (see Type_aberration value)
    # -----------------------------------------------------------------------------
    if Type_aberration[0] is 'Z':
        for z in range(N_modes):
            phase_array[z,:]*=0.
            if (z == int(Type_aberration[1])+12-4) is True:
                phase_array[z,:]=modes_zern[z,:]*RMS

    if Type_aberration is 'NCPA':
        phase_array[0:12,:]*=0.
            
 
    total_phase_map     = np.sum(phase_array,0)
    phase_map           = total_phase_map.reshape(size_pup,size_pup)
    phase_map_rotated   = iof.cent_rot(phase_map,rot_angle,np.array([size_pup/2.,size_pup/2.]))

    

    if display is True:
        plt.figure()
        im=plt.imshow(phase_map_rotated,cmap='plasma')
        plt.clim([-np.pi,np.pi])
        plt.colorbar(im,orientation='vertical')
        plt.axis('off')
        plt.title('Phase_map_'+str(RMS)+'rad-rms_'+str(N_modes)+'modes')
        plt.show()
    if save is True:
        date=datetime.datetime.today().strftime('%Y-%m-%d')
        clock=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_save='/mnt/md0/svievard/Data/DM_phase_screens/'+date+'/'
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
        path_save=folder_save+clock+'_'
        plt.savefig(path_save+'dm-map_'+save_name+'.png')
        hdu = fits.PrimaryHDU(phase_map_rotated)
        hdu.writeto(path_save+'dm-map_'+save_name+'.fits',overwrite=True)

    return phase_map_rotated
   
def reshape_to_real_dmmap(dmmap, rotation = True):
    # Vincent: make rotation optional - todo: any given angle

    dmmap_current            = dmmap

    ##### Info pupil
    rot_angle               = (0.0, 6.25)[rotation]                                  # In degrees
    size_pup_dm             = 44
    size_dm                 = 50
    dmmap_real              = np.zeros([size_dm,size_dm]) 

    ##### Get real pupil mask
    # Vincent: amend for scexao5
    real_pupil_dm           = fits.getdata('/mnt/sdata00/svievard/Telescope_pupil/Subaru/pupil_on_dm.fits')
    #real_pupil_dm           = fits.getdata('/media/data/Sebviev/LWE/data/Modes/pupil_on_dm.fits')
    size_pup                = real_pupil_dm.shape[1]

    ##### Rotate real pupil mask
    real_pupil_dm_rotated   = iof.cent_rot(real_pupil_dm,rot_angle,np.array([size_pup/2.,size_pup/2.]))

    ##### Resample to actual size of the pupil
    dmmap_resize            = iof.Fourier_resample(real_pupil_dm_rotated,[size_pup_dm,size_pup_dm])
    dmmap_resize[np.where(dmmap_resize < 0.5)] = 0.
    dmmap_resize[np.where(dmmap_resize > 0.5)] = 1.

    ##### Immerge the real pupil into the DM map
    dmmap_real[1:size_pup_dm+1,2:size_pup_dm+2] = dmmap_resize


    ##### Multiply given dmmap by the mask

    if len(np.shape(dmmap_current)) == 2:
        dmmap_current = dmmap_current*dmmap_real

    if np.shape(dmmap_current) > 2:
        size_cube = np.shape(dmmap_current)[0]
        for i in range(size_cube):
            dmmap_current[i,:,:] = dmmap_current[i,:,:]*dmmap_real


    return dmmap_current
     
############### TALK TO THE SCEXAO DM

def apply_dm_phasemap(phasemap, rms_value, wavelength_microns = 0.750, dmdisp= "dm00disp06", display=False, save = False, save_name = 'Test'):

    # ----------------------------------------------------------------------------
    # Initialize shared memory
    # ----------------------------------------------------------------------------
    dm=shm(dmdisp)
    #dm=shm("dm00disp08")
    
    # ----------------------------------------------------------------------------
    # Initialize size of pupil and DM map
    # ----------------------------------------------------------------------------
    size_pup_dm  = 44  # Size of the pupil on the DM
    size_dm      = 50  # Size of the DM
    dm_command   = np.zeros([size_dm,size_dm])  


    # ----------------------------------------------------------------------------
    # Resample the map, in case it is not the right size
    # ----------------------------------------------------------------------------
    if phasemap.shape[0] != size_pup_dm :
        print('WARNING : Resampling of the provided phase map to fit the DM')
        phasemap_dm = iof.Fourier_resample(phasemap,[size_pup_dm,size_pup_dm])
    else:
        phasemap_dm = phasemap

    # ----------------------------------------------------------------------------
    # Fill the dm command image + convert into microns
    # ----------------------------------------------------------------------------
    rms_microns=rad2microns(rms_value,wavelength_microns)
    dm_command[1:size_pup_dm+1,2:size_pup_dm+2]=rad2microns(phasemap_dm,wavelength_microns)

    if display is True:
        plt.figure()
        im=plt.imshow(dm_command,cmap='plasma')
        cmin=rad2microns(-np.pi,wavelength_microns)
        cmax=rad2microns(+np.pi,wavelength_microns)
        plt.clim([cmin,cmax])
        plt.colorbar(im,orientation='vertical')
        plt.axis('off')
        plt.title('Phase_map_'+str(np.around(rms_microns,decimals=3))+'microns-rms')
        plt.show()

    if save is True:
        date=datetime.datetime.today().strftime('%Y-%m-%d')
        clock=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_save='/mnt/md0/svievard/Data/DM_shape/'+date+'/'
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
        path_save=folder_save+clock+'_'
        # plt.savefig(path_save+'Phase_map_'+str(np.around(rms_microns,decimals=3))+'microns-rms.png')
        hdu = fits.PrimaryHDU(dm_command)
        hdu.writeto(path_save+'Phase_map_DM_'+save_name+'.fits')

    dm_command /= 2. #### DM in reflexion...so 2 coeff.

    dm.set_data(dm_command.astype(np.float32))
    # time.sleep(0.001)

    # -----------------------------------------------------------------------------
    # Close shared memory
    # -----------------------------------------------------------------------------

    dm.close()

    return dm_command

def get_DM_shape(dmdisp= "dm00disp06", display=False, save=False, save_name='Test'):
    # ----------------------------------------------------------------------------
    # Initialize shared memory
    # ----------------------------------------------------------------------------
    dm=shm(dmdisp)

    
    # Image acquisition
    image_dm = dm.get_data(True, True, timeout = 1.).astype(float)
    
    if display is True:
        plt.figure(0)
        plt.imshow(image_dm,aspect='auto')
    if save is True:
        date=datetime.datetime.today().strftime('%Y-%m-%d')
        clock=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_save='/mnt/md0/svievard/Data/DM_shape/'+date+'/'
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
        path_save=folder_save+clock+'_'
        hdu = fits.PrimaryHDU(image_dm)
        hdu.writeto(path_save+'_DMshape_'+save_name+'.fits',overwrite=True)
    # -----------------------------------------------------------------------------
    # Close camera and Dark shared memory
    # -----------------------------------------------------------------------------
    dm.close()

    return 

##### UNIT CONVERSION

def rad2microns(array,wavelength):
    # Converts radians into microns, depending on the selected wavelength 
    #
    # Wavelength should be in microns

    phase_microns=(array*wavelength)/(2*np.pi)

    return phase_microns


