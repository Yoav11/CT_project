import numpy as np
from attenuate import *
from ct_calibrate import *

def hu(p, material, reconstruction, scale):
	""" convert CT reconstruction output to Hounsfield Units
	calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy p and scale given."""

	# use result to convert to hounsfield units
	reconstruction = (reconstruction - 1) * 1000 

	# limit minimum to -1024, which is normal for CT data.
	reconstruction[reconstruction < -1024] = -1024

	return reconstruction