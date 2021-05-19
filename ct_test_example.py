
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

	# plt.plot(p[63, :])
	save_plot(y[63,:], 'results', 'test_1_plot')

def test_2():
	# Tests that reconstructed implant image has correct attenuation values

	# set up initial conditions
	source_energy = 0.12

	p = ct_phantom(material.name, 256, 3)
	s = fake_source(material.mev, source_energy, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# Find location of implant using phantom reference
	implant_location = np.where(p == 7)
	# Average attenuation values on reconstructed image
	implant_attenuation = np.mean(y[implant_location])
	# Find material attenuation coefficients for chosen ideal source
	material_attenuations = material.coeffs[:, np.where(material.mev == source_energy)[0][0]]

	# Find the difference between the material coefficients and the computed average and find the min
	error = np.abs(material_attenuations - implant_attenuation)
	material_idx = error.argmin()

	# Use the closest material attenuation value to infer material of implant
	predicted_material = material.name[material_idx]

	# save some meaningful results
	f = open('results/test_2/test_2_output.txt', mode='w')
	f.write(f"Implent attenuation value is  {round(implant_attenuation, 3)} \n")
	f.write(f"Predicted material is {predicted_material} \n")
	f.write(f"Attenuation error is {round(error[material_idx], 3)} \n")
	f.close()

	# Set parameters for drawing rectangle
	x1 = min(implant_location[1])
	y1 = implant_location[0][0]

	lx = implant_location[0][-1] - y1
	ly =  max(implant_location[1]) - min(implant_location[1])

	# Plot rectangle around implant and label using predicted material
	save_rectangle(y, 'results/test_2', 'test_2_image', (x1, y1), (lx, ly), predicted_material)

	# Check predicted material for implant is Titanium
	assert predicted_material == "Titanium", f"Implant should be Titanium, got {predicted_material}"
	# Check attenuation error is within bounds
	assert error[material_idx] < 0.5, f"Attenuation error too large, got {error[material_idx]}"

def test_3():
	# Tests that reconstructed image is accurate to the original phantom

	# set up initial conditions
	source_energy = 0.14

	p = ct_phantom(material.name, 256, 7)
	s = fake_source(source.mev, source_energy, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.1, 256)

	# Find material attenuation coefficients for chosen ideal source
	material_attenuations = material.coeffs[:, np.where(material.mev == source_energy)[0][0]]
	# Convert phantom image from indices to true attenuation values
	p = convert_phantom(p, material_attenuations)

	# Find the error between the scaled phantom and reconstructed image
	error_image = y-p
	# Measure RMS error, ignoring pixels outside of scanning circle
	rms_error = np.sqrt(np.mean(error_image[np.where(error_image > -1)]**2))

	# save some meaningful results
	save_draw(y, 'results/test_3', 'test_3_image')
	save_draw(p, 'results/test_3', 'test_3_phantom_scaled')
	save_draw(error_image, 'results/test_3', 'test_3_error')

	f = open('results/test_3/test_3_output.txt', mode='w')
	f.write(f"RMS reconstruction error is {round(rms_error, 5)} \n")
	f.close()

	# Check that measured RMS error is within acceptable bounds
	assert rms_error < 0.06, f"RMS error too large, got {rms_error}"


# Run the various tests
# print('Test 1')
# test_1()
print('Test 2')
test_2()
print('Test 3')
test_3()
