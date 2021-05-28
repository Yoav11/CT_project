
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

# create object instances
material = Material()
source = Source()

# define each end-to-end test here, including comments
# these are just some examples to get you started
# all the output should be saved in a 'results' directory

def test_1():
	''' 
	Test for reconstruction geometry:

	reconstruct a point attenuator phantom away from the origin,
	check the location of the attenuator matches the phantom and 
	evaluate the radius over which the attenuation value spreads
	'''

	# INITIAL CONDITIONS

	# point attenuator phantom used for comparing geometry
	p = ct_phantom(material.name, 256, 2, 'Titanium')

	# ideal source used for evaluating error and spread
	source_energy = 0.12
	s = fake_source(material.mev, source_energy, method='ideal')

	# SETUP

	# generate scan
	scan = scan_and_reconstruct(s, material, p, 0.1, 256)

	# find material attenuation coefficients for chosen ideal source
	material_attenuations = material.coeffs[:, np.where(material.mev == round(0.7*source_energy,3))[0][0]]
	# convert phantom image from indices to true attenuation values
	p = convert_phantom(p, material_attenuations)
	
	# find location of the pixel with maximum value in the reconstructed image
	max_index = np.unravel_index(scan.argmax(), scan.shape)
	# find absolute error
	error = abs(scan - p)

	# measure radius of spread on error image
	radius = measure_spread(error, max_index, 0.01)

	# TESTS

	# save diagram with highlighted attenuation area
	save_circle(scan, 'results/test_1', 'test_1_image', max_index, radius, 'Error')

	# find location of the pixel with maximum value in the phantom image
	max_index_phantom = np.unravel_index(p.argmax(), p.shape)

	# save location of maximum value and area of attenuation circle
	area = np.pi*(radius**2)
	f = open('results/test_1/test_1_output.txt', mode='w')
	f.write(f"Position of max attenuation, Scan: {max_index}, Phantom: {max_index_phantom} \n")
	f.write(f"Area of circle { round(area, 3) }")
	f.close()

	# expect that location of maximum attenuation in reconstructed image matches location in phantom
	assert max_index == max_index_phantom, f'max attenuation location does not \
						match, got {max_index}, expected {max_index_phantom}'
	# expect that attenuation spread area is sufficiently small
	assert area < 40, f'area is above threshold, got {area}'

def test_2():
	''' 
	Test for reconstruction values:

	reconstruct a hip implant phantom, check that the attenuation 
	value of the implant most closely matches the attenuation value 
	of the phantom implant material and evaluate the attenuation error of the implant
	'''

	# INITIAL CONDITIONS

	# hip implant phantom
	p = ct_phantom(material.name, 256, 3)
	# ideal source used for comparing material values
	source_energy = 0.10
	s = fake_source(material.mev, source_energy, method='ideal')

	# SETUP

	# generate scan
	scan = scan_and_reconstruct(s, material, p, 0.01, 256)

	# find location of implant using phantom as reference
	implant_location = np.where(p == 7)
	# average attenuation values of implant on reconstructed image
	implant_attenuation = np.mean(scan[implant_location])
	# find material attenuation coefficients for chosen ideal source
	material_attenuations = material.coeffs[:, np.where(material.mev == round(0.7*source_energy,3))[0][0]]

	# find the difference between the material coefficients and the computed average and find the min
	error = np.abs(material_attenuations - implant_attenuation)
	material_idx = error.argmin()

	# use the closest material attenuation value to infer material of implant
	predicted_material = material.name[material_idx]

	# Set parameters for drawing rectangle
	x1 = min(implant_location[1])
	y1 = implant_location[0][0]

	lx = implant_location[0][-1] - y1
	ly =  max(implant_location[1]) - min(implant_location[1])

	# TESTS

	# save diagram with highlighted implant location and material
	save_rectangle(scan, 'results/test_2', 'test_2_image', (x1, y1), (lx, ly), predicted_material)
	
	# save implant attenuation value, error and inferred material
	f = open('results/test_2/test_2_output.txt', mode='w')
	f.write(f"Implant attenuation value is  {round(implant_attenuation, 3)} \n")
	f.write(f"Predicted material is {predicted_material} \n")
	f.write(f"Attenuation error is {round(error[material_idx], 3)}\n")
	f.write(f"As a percentage is {round(error[material_idx]*100/implant_attenuation, 3)} %\n")
	f.close()

	# Expect predicted material for implant to be Titanium
	assert predicted_material == "Titanium", f"Implant should be Titanium, got {predicted_material}"
	# Expect attenuation error is sufficiently small
	assert error[material_idx] < 0.05, f"Attenuation error too large, got {error[material_idx]}"

def test_3():
	''' 
	Test for reconstruction accuracy:

	reconstruct the pelvic fixation pins phantom, check that the error
	between the phantom and the reconstruction is sufficiently small
	'''

	# INITIAL CONDITIONS
	
	# pelvic fixation pins phantom
	p = ct_phantom(material.name, 256, 7)
	# ideal source used for converting phantom to predictable attenuation value
	source_energy = 0.14
	s = fake_source(source.mev, source_energy, method='ideal')

	# SETUP

	# generate scan
	scan = scan_and_reconstruct(s, material, p, 0.1, 256)

	# Find material attenuation coefficients for chosen ideal source
	material_attenuations = material.coeffs[:, np.where(material.mev == round(0.7*source_energy,3))[0][0]]
	# Convert phantom image from indices to true attenuation values
	p = convert_phantom(p, material_attenuations)

	# Find the error between the scaled phantom and reconstructed image
	error_image = scan-p
	# Measure RMS error, ignoring pixels outside of scanning circle
	rms_error = np.sqrt(np.mean(error_image[np.where(error_image > -1)]**2))

	# TESTS

	# save scan, scaled phantom and error images
	save_draw(scan, 'results/test_3', 'test_3_image')
	save_draw(p, 'results/test_3', 'test_3_phantom_scaled')
	save_draw(error_image, 'results/test_3', 'test_3_error')

	# save RMS reconstruction error
	f = open('results/test_3/test_3_output.txt', mode='w')
	f.write(f"RMS reconstruction error is {round(rms_error, 5)} \n")
	f.close()

	# expect that measured RMS error is sufficiently small
	assert rms_error < 0.06, f"RMS error too large, got {rms_error}"

def test_4():
	energies = [0.005, 0.01, 0.015, 0.02, 0.03, 0.1, 0.12, 0.15]
	for s in energies:
		source = fake_source(material.mev, s, method='ideal')
		y = ct_detect(source, material.coeff('Water'),np.arange(0, 10.1, 0.1), 1)
		plt.plot(np.log(y), label=round(s*1000, 3))
	plt.xlabel('depth')
	plt.ylabel('residual energy')
	plt.legend()

	full_path = get_full_path('results', 'test_4')
	plt.savefig(full_path)
	plt.close()

def test_5():
	s1 = fake_source(material.mev, 0.08, material.coeff('Aluminium'), 2)
	s2 = fake_source(material.mev, 0.08, material.coeff('Aluminium'), 0)

	y = ct_detect(s1, material.coeff('Water'),np.arange(0, 10.1, 0.1), 1)
	plt.plot(np.log(y), label='2mm Aluminium')

	y = ct_detect(s2, material.coeff('Water'),np.arange(0, 10.1, 0.1), 1)
	plt.plot(np.log(y), label='non-filtered')

	plt.xlabel('depth')
	plt.ylabel('residual energy')
	plt.legend()

	full_path = get_full_path('results', 'test_4')
	plt.savefig(full_path)
	plt.close()

def test_6():
	s1 = source.photon('100kVp, 2mm Al')


	y = ct_detect(s1, material.coeff('Water'),np.arange(0, 10.1, 0.1), 1)
	plt.plot(np.log(y), label='Water')

	y = ct_detect(s1, material.coeff('Titanium'),np.arange(0, 10.1, 0.1), 1)
	plt.plot(np.log(y), label='Titanium')

	y = ct_detect(s1, material.coeff('Bone'),np.arange(0, 10.1, 0.1), 1)
	plt.plot(np.log(y), label='Bone')

	plt.xlabel('depth')
	plt.ylabel('residual energy')
	plt.legend()

	full_path = get_full_path('results', 'test_4')
	plt.show()
	plt.savefig(full_path)
	plt.close()

# Run the various tests
# print('Test 1')
# test_1()
# print('Test 2')
# test_2()
# print('Test 3')
# test_3()
print('Test 3')
test_3()
# test_6()
