
# these are the imports you are likely to need
import numpy as np
from numpy.lib.npyio import save
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *
import matplotlib.pyplot as plt

# create object instances
material = Material()
source = Source()

# define each end-to-end test here, including comments
# these are just some examples to get you started
# all the output should be saved in a 'results' directory

def test_1():
	''' 
	Test for reconstruction geometry: reconstruct a point attenuator phantom and work out the radius over which the attenuation value spreads
	'''
	#Create phantom for point attenuator and produce reconstructed image
	p = ct_phantom(material.name, 256, 2, 'Titanium')
	s = fake_source(material.mev, 0.12, material.coeff('Aluminium'), 2, method='ideal')
	scan = scan_and_reconstruct(s, material, p, 0.1, 256)

	#Replace values in phantom with known attenuation
	p[p==7]=material.coeffs[:, np.where(material.mev == 0.12)[0][0]][7]
	
	#Find absolute error
	error = abs(scan - p)
	
	#Find location of the pixel with maximum value in the reconstructed image
	max_index = np.unravel_index(scan.argmax(), scan.shape)

	#Initialise loop with given values
	last_sum = 0
	avg_error = 1
	
	#Iterate over the pixels surrouding the maximum value and calculates the average error at each 'layer' - maximum error is 0.01
	for n in range(1,256):
		if avg_error < 0.01:
			break
		layer = (error[(max_index[0]-n):(max_index[0]+n+1), (max_index[1]-n):(max_index[1]+n+1)])
		avg_error = (np.sum(layer)-last_sum)/ (((2*n +1)**2) - (2*n-1)**2)
		last_sum = np.sum(layer)
	
	#Draw circle at required radius
	base_radius=np.sqrt(2)/2 
	radius=base_radius + np.sqrt(2)*(n-2)
	
	draw_circle(scan, max_index, radius, 'error' )

	#Save diagram and area
	area=np.pi*(radius**2)
	f = open('results/test_1/test_1_output.txt', mode='w')
	f.write(f"area for set error {area}")
	f.close()

	save_circle(scan, 'results/test_1', 'test_1_output', max_index, radius, 'Error')

	assert area< 500, f'Area is above  BLAH BLAH, got {area}'

def test_2():
	# explain what this test is for

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 2)
	s = source.photon('80kVp, 1mm Al')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# save some meaningful results
	save_plot(y[128,:], 'results', 'test_2_plot')

	# how to check whether these results are actually correct?

def test_3():
	# explain what this test is for

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 1)
	s = fake_source(source.mev, 0.1, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.1, 256)

	# save some meaningful results
	f = open('results/test_3_output.txt', mode='w')
	f.write('Mean value is ' + str(np.mean(y[64:192, 64:192])))
	f.close()

	# how to check whether these results are actually correct?



# Run the various tests
#print('Test 1')
test_1()


# print('Test 2')
# test_2()
# print('Test 3')
# test_3()
