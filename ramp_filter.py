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

	# set up filter to be at least twice as long as input
	m = np.ceil(np.log(2*n-1) / np.log(2))
	m = int(2 ** m)

	print('Ramp filtering')

	# initialise frequency array and set max frequency to nyquist frequency
	f = np.fft.fftfreq(m)
	f_max = 1 / (2 * scale)

	# digital correction
	f[0] = f[1]/6

	# Ram-Lak filter with raised_cosine
	filter = f_max * np.abs(f) * np.power(np.cos(np.pi/2 * f / f_max), alpha)
	# compute fft of sinogram
	freq_distribution = np.fft.fft(sinogram, m, axis=1)
	
	# filter implementation
	freq_distribution *= filter

	# compute filtered sinogram
	sinogram = np.real(np.fft.ifft(freq_distribution, axis=1))[:, :n]

	return sinogram