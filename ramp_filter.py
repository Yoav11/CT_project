import math
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

def ramp_filter(sinogram, scale, alpha=0.001):
	""" Ram-Lak filter with raised-cosine for CT reconstruction

	fs = ramp_filter(sinogram, scale) filters the input in sinogram (angles x samples)
	using a Ram-Lak filter.

	fs = ramp_filter(sinogram, scale, alpha) can be used to modify the Ram-Lak filter by a
	cosine raised to the power given by alpha."""

	# get input dimensions
	angles = sinogram.shape[0]
	n = sinogram.shape[1]

	#Set up filter to be at least twice as long as input
	m = np.ceil(np.log(2*n-1) / np.log(2))
	m = int(2 ** m)
	print(angles, n, m)

	# apply filter to all angles
	print('Ramp filtering')

	filt = np.zeros((angles, m))
	f_max = 1 / (scale)
	f = np.fft.fftfreq(m)
	f[0] = f[1]/6
	fourier_filter = (1/(2*scale)) * np.abs(f) * np.power(np.cos(np.pi/2 * f / f_max), alpha)
	filt = np.fft.fft(sinogram, m, axis=1)
	
	filt *= fourier_filter

	sinogram = np.real(np.fft.ifft(filt, axis=1))[:, :n]
	return sinogram