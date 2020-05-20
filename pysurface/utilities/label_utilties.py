
import numpy as np


def region_indices(lab_obj, regions):
	"""
	Parameters:
	- - - - -
	lab_obj : GiftiImage
	label object

	"""

	cdata = lab_obj.darrays[0].data

	lt = lab_obj.get_labeltable().get_labels_as_dict()
	reg2lab = dict(zip(map(str, lt.values()), lt.keys()))

	indices = []

	for r in regions:
		indices.append(np.where(cdata == reg2lab[r])[0])

	indices = np.concatenate(indices)

	return indices

def toFeatures(parcels):

	"""
	Converts a parcellation into a binary array of features.

	Parameters:
	- - - - -
	parcels: int, array
		cortical parcellation map

	Returns:
	- - - -
	features: int, array (N x p)
		binary feature array of N samples by p unique parcels
	"""

	unique = list(set(list(np.unique(parcels))).difference({0,-1}))

	features = np.zeros((parcels.shape[0], len(unique)))
	
	for j, u in enumerate(unique):

		features[:,j] = (parcels == u).astype(np.int32)
	
	return features

