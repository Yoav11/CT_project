import numpy as np
from attenuate import *
from ct_calibrate import *

def hu(p, material, reconstruction, scale):
	""" convert CT reconstruction output to Hounsfield Units
	calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy p and scale given."""

	# use water to calibrate
	u = ct_detect(p, material.coeff('Water'), 1)

	# put this through the same calibration process as the normal CT data
	n = max(reconstruction.shape)
	n = float(2*n)
	depth = n*scale

	a = ct_detect(p, material.coeff('Air'), depth)
	u = -np.log(u/a)

	# use result to convert to hounsfield units
	reconstruction = (reconstruction - u) * 1000 / u

	# limit minimum to -1024, which is normal for CT data.
	reconstruction[reconstruction < -1024] = -1024

	return reconstruction