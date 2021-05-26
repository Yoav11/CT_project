import numpy as np
import scipy
from scipy import interpolate
import sys
from ct_detect import ct_detect

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

	return sinogram