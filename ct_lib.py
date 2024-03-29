import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import os
from matplotlib.patches import Rectangle

def draw(data, map='gray', caxis=None):
	"""Draw an image"""
	create_figure(data, map, caxis)
	plt.show()


def plot(data):
	"""plot a graph"""
	plt.plot(data)
	plt.show()

def draw_rectangle(data, coordinates, lengths, label, map='gray', caxis=None):
	"""Draw an image with a highlighted rectangle"""
	create_rectangle_figure(data, coordinates, lengths, label)
	plt.show()

def save_rectangle(data, storage_directory, file_name, coordinates, lengths, label, map='gray', caxis=None):
	"""Save an image with a highlighted rectangle"""
	create_rectangle_figure(data, coordinates, lengths, label)
	
	full_path = get_full_path(storage_directory, file_name)
	plt.savefig(full_path)
	plt.close()

def save_draw(data, storage_directory, file_name, map='gray', caxis=None):
	"""save an image"""
	create_figure(data, map, caxis)

	full_path = get_full_path(storage_directory, file_name)
	plt.savefig(full_path)
	plt.close()

def save_plot(data, storage_directory, file_name):
	"""save a graph"""
	full_path = get_full_path(storage_directory, file_name)
	plt.plot(data)
	plt.savefig(full_path)
	plt.close()

def convert_phantom(phantom, attenuations):
	"""Convert phantom from index values to attenuation values"""
	for i in range(len(attenuations)):
		phantom[np.where(phantom == i)] = attenuations[i]
	return phantom

def save_numpy_array(data, storage_directory, file_name):
	"""save a numpy array in .npy format"""

	full_path = get_full_path(storage_directory, file_name)

	np.save(full_path, data)

def load_numpy_array(storage_directory, file_name):
	"""load a .npy file into numpy array"""

	full_path = os.path.join(storage_directory, file_name)

	#add .npy extension if needed
	if not full_path.endswith('.npy'):
		full_path = full_path + '.npy'

	if not os.path.exists(full_path):
		raise Exception('File named ' + full_path + ' does not exist')

	return np.load(full_path)

def get_full_path(storage_directory, file_name):
	#create storage_directory if needed
	if not os.path.exists(storage_directory):
		os.makedirs(storage_directory)

	full_path = os.path.join(storage_directory, file_name)

	return full_path

def create_figure(data, map, caxis = None):
	fig, ax = plt.subplots()

	plt.axis('off') # no axes

	if caxis is None:
		im = plt.imshow(data, cmap=map)
	else:
		im = plt.imshow(data, cmap=map, vmin=caxis[0], vmax=caxis[1])

	# equal aspect ratio
	ax.set_aspect('equal', 'box')
	plt.tight_layout()

	#add colorbar
	plt.colorbar(im, orientation='vertical')

def create_circle_figure(data, coordinates, radius, label):
	"""Create plot with highlighted circle"""
	fig, ax = plt.subplots()

	plt.axis('off') # no axes

	im = plt.imshow(data, cmap='gray')

	# equal aspect ratio
	ax.set_aspect('equal', 'box')
	plt.tight_layout()

	circle1 = plt.Circle((coordinates[1], coordinates[0]), radius, label=label, color='r', fill=False)
	ax.add_patch(circle1)

	plt.colorbar(im, orientation='vertical')
	plt.legend()
  
def create_rectangle_figure(data, coordinates, lengths, label):
	"""Create plot with highlighted rectangle"""
	fig, ax = plt.subplots()

	plt.axis('off') # no axes

	ax.set_aspect('equal', 'box')
	im = plt.imshow(data, cmap='gray')

	# equal aspect ratio
	ax.add_patch( Rectangle((coordinates[0], coordinates[1]), lengths[0], lengths[1], fc ='none', ec ='r', label=label))
	plt.tight_layout()
	
	plt.colorbar(im, orientation='vertical')
	plt.legend()

def draw_circle(data, coordinates, radius, label, map='gray', caxis=None):
	"""Draw an image with a highlighted circle"""
	create_circle_figure(data, coordinates, radius, label)
	plt.show()

def save_circle(data, storage_directory, file_name, coordinates, radius, label, map='gray', caxis=None):
	"""Save an image with a highlighted circle"""
	create_circle_figure(data, coordinates, radius, label)
	full_path = get_full_path(storage_directory, file_name)
	
	plt.savefig(full_path)
	plt.close()

def measure_spread(image, location, threshold=0.01):
	"""given an error image and location of a point, returns radius of circle around that point
	where error is above a given threshold"""

	# iterate over the pixels surrouding the maximum value and 
	# calculate the average error at each 'layer' until error is small
	last_sum = 0
	for n in range(1,256):
		# select pixels around the maximum value at distance n
		layer = (image[(location[0]-n):(location[0]+n+1), (location[1]-n):(location[1]+n+1)])
		# compute the average error on the outer pixels of that layer
		avg_error = (np.sum(layer)-last_sum)/ ((np.power((2*n+1), 2)) - np.power((2*n-1), 2))
		# keep track of current error sum for next iteration
		last_sum = np.sum(layer)

		# stop when average error drops below threshold
		if avg_error < threshold:
			break

	# calculate radius around spreaded area, if n = 0, draws around point attenuator
	base_radius = np.sqrt(2)/2
	return base_radius + np.sqrt(2)*(n-1)
