import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
import sys
from ct_detect import ct_detect
from ct_lib import *

def ct_calibrate(photons, material, sinogram, scale, correct=True):

	""" ct_calibrate convert CT detections to linearised attenuation
	sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
	in x (angles x samples) and returns a linear attenuation sinogram
	(angles x samples). photons is the source energy distribution, material is the
	material structure containing names, linear attenuation coefficients and
	energies in mev, and scale is the size of each pixel in x, in cm."""
	# Get dimensions and work out detection for just air of twice the side
	# length (has to be the same as in ct_scan.py)

	# get sinogram dimensions
	n = sinogram.shape[1]
	angles = sinogram.shape[0]

	print('Calibrating')

	# initialise calibration sinogram
	calibration_scan = np.zeros((angles, n))


	# initialise depth array to twice the side length and scale
	depth = np.full(n, float(2*n))
	depth *= scale

	# run scan through air once and repeat along all angles
	calibration_scan = np.repeat( ct_detect(photons, material.coeff('Air'), depth)[:, np.newaxis], \
					angles, axis=1 )

	calibration_scan = np.transpose(calibration_scan)

	# perform calibration
	sinogram = -np.log(sinogram/calibration_scan)

	
	

	#BEAM HARDENING CORRECTION 

	#create array of water depths and find attenuations at each depth
	water_depth=np.linspace(0, 2*n, n)*scale

	calibration_air=ct_detect(photons, material.coeff('Air'), depth)
	water_attenuation=ct_detect(photons, material.coeff('Water'), water_depth)
	water_calibrated = -np.log(water_attenuation/calibration_air)

	#interpolate to obtain thickness as a function of attenuation only for the linear region
	f_linear=scipy.interpolate.interp1d(water_calibrated, water_depth, 'linear', fill_value='extrapolate')

	#calibrate with respect to water by substituting attenuation values in original sinogram with corresponding water thickness.
	#if saturation is reached, substitute with saturation value
	
	sinogram=f_linear(sinogram)
	
	return sinogram