
# these are the imports you are likely to need
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# create object instances
material = Material()
source = Source()

# define each end-to-end test here, including comments
# these are just some examples to get you started
# all the output should be saved in a 'results' directory

def test_1():
	# explain what this test is for

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 2, 'Titanium')
	s = fake_source(material.mev, 0.12, material.coeff('Aluminium'), 2, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.1, 256)

	# save some meaningful results
	save_draw(y, 'results', 'test_1_image')
	save_draw(p, 'results', 'test_1_phantom')
	save_draw(y-p, 'results', 'test_1_error')

	# how to check whether these results are actually correct?
	
	error = y - p
	max_index = np.unravel_index(y.argmax(), y.shape)

	f = open('results/test_1_output.txt', mode='w')
	f.write(f"max index  {max_index}")
	f.close()

	plt.plot(p[63, :])
	save_plot(y[63,:], 'results', 'test_1_plot')

	

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

def test_4():
	#Plot a 3D surface of a point
	p = ct_phantom(material.name, 256, 2, 'Titanium')
	s = fake_source(material.mev, 0.12, material.coeff('Aluminium'), 2, method='ideal')
	scan = scan_and_reconstruct(s, material, p, 0.1, 256)

	#Find average error of each layer around point attenuator
	max_index = np.unravel_index(scan.argmax(), scan.shape)

	scan /= np.max(scan)
	scan *= 7
	'''draw(scan)'''
	error = abs(scan - p)
	'''draw(error)'''

	last_sum = 0
	avg_error = 1
	
	for n in range(1,10):
		if avg_error < 0.1:
			break
		layer = (error[(max_index[0]-n):(max_index[0]+n+1), (max_index[1]-n):(max_index[1]+n+1)])
		avg_error = (np.sum(layer)-last_sum)/ (((2*n +1)**2) - (2*n-1)**2)
		last_sum = np.sum(layer)
		print(avg_error)

	#Plot on 3D axis
	x_axis = np.linspace(0, 256, 256)
	y_axis = np.linspace(0, 256, 256)

	X, Y = np.meshgrid(x_axis, y_axis)
	Z1 = atten_values
	Z2 = p
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot_surface(X, Y, Z1, rstride=1, cstride=1,
			 edgecolor='none')
	#ax.plot_surface(X, Y, Z2, rstride=1, cstride=1,
			 #edgecolor='none')
	#plt.show()


	#print(np.nonzero(p))
	#print(np.nonzero(atten_values))

# Run the various tests
''' print('Test 1')
test_1()
print('Test 2')
test_2()
print('Test 3')
test_3() '''
print('Test 4')
test_4()

#Find the max index
#max_index = np.unravel_index(y.argamax(), y.shape)