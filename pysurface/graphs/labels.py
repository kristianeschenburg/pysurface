import networkx as nx
import numpy as np

from . import adjacency


class BoundaryMap(object):

    """
    Class to find vertices that exist on the boundary of two regions.

    Parameters:
    - - - - -
    index_map: dictionary
        mapping of region names to indices
    
    adj_list : dictionary
        adjacency list for surface mesh on which label map lives
    """

    def __init__(self, index_map, adj_list):

        self.index_map = index_map
        self.adj_list = adj_list

    def find_boundaries(self):

        """
        Method to identify vertices that exist at the boundary of two regions.
        """

        boundaries = {k: None for k in self.index_map.keys()}

        for region, inds in self.index_map.items():

            binds = []

            for tidx in inds:

                neighbors = set(self.adj_list[tidx])
                outer = neighbors.difference(set(inds))
                if len(outer) > 0:
                    binds.append(tidx)
            
            boundaries[region] = binds

        self.boundaries = boundaries


class Components(adjacency.SurfaceAdjacency):

    """
    Generate connected-components from surface adjacency object.

    Parameters:
    - - - - -
    vertices: array, float
        points of surface mesh
    faces: array, int
        triangles of surface mesh
    labels: 
    """

    def __init__(self, vertices, faces, surf_map, map_type='parcels'):

        """
        Initialize the connected components object.
        """

        self.surf_map = surf_map
        self.vertices = vertices
        self.faces = faces
        self.map_type=map_type

    def generate(self, indices=None):
        
        """
        Method to create surface adjacency list.
        """

        # Get faces attribute
        faces = self.faces.tolist()
        accepted = np.zeros((self.vertices.shape[0]))

        # get indices of interest
        if not np.any(indices):
            indices = list(np.unique(np.concatenate(faces)))
        indices = np.sort(indices)

        # create array of whether indices are included
        # cancels out search time
        accepted[indices] = 1
        accepted = accepted.astype(bool)

        # Initialize adjacency list
        adj = {k: [] for k in indices}

        # loop over triangles in mesh
        for face in faces:

            # loop over triangles in face
            for j, vertex in enumerate(face):
                idx = (np.asarray(face) != vertex)

                # check if vertex is in indices, and which of its neighbors have the same
                # label
                if accepted[vertex]:
                    nbs = np.asarray([n for n in np.asarray(face)[idx] if accepted[n]]).astype(np.int32)

                    if self.map_type == 'parcel':
                        adj[face[j]].append(nbs[self.surf_map[vertex] == self.surf_map[nbs]])
                    elif self.map_type == 'metric':
                        adj[face[j]].append(nbs[ (self.surf_map[nbs] != 0)])

        for k in adj.keys():
            if adj[k]:
                adj[k] = list(set(np.concatenate(adj[k])))

        # Set adjacency list field
        self.adj = adj

    def components(self, size=None):

        """
        Compute the number of connected components.
        """

        G = nx.from_dict_of_lists(self.adj)
        comp_generator = nx.connected_components(G)

        components = {}
        for j, c in enumerate(comp_generator):
            components[j] = list(c)

        if size:
            to_delete = []
            for c, idx in components.items():
                if len(idx) < size:
                    to_delete.append(c)
            
            for c in to_delete:
                del(components[c])

        self.components_ = components
        

class LabelAdjacency(object):

    """
    Generates adjacency list of labels in a parcellation.  For given parcel K,
    computes labels of the parcels neighboring parcel K.

    Requires as input a surface adjacency list.

    Parameters:
    ------------
        label : np.array
            vector of vertex label assignments
        surfAdj : dict
            surface adjacency list
    """

    def __init__(self, label, adjacency):

        self.label = label
        self.adjacency = adjacency

    def generate(self):

        """

        """

        labels = self.label
        adjacency = self.adjacency

        # get unique non-midline labels in cortical map
        labs = set(labels).difference({0, -1})

        lab_adj = {}.fromkeys(labs)

        # Loop over unique values in label map
        for k in labs:

            # Find vertices belonging to parcel with current label

            tempInds = np.where(labels == k)[0]

            neighbor_labels = []

            # Loop over vertices in each parcel
            for j in tempInds:

                neighbor_labels.append(labels[adjacency[j]])
            
            neighbor_labels = set(np.concatenate(neighbor_labels))
            neighbor_labels = [l for l in neighbor_labels if l not in [k, 0, -1]]
            lab_adj[k] = neighbor_labels

        self.adj = lab_adj


class TPM(object):

    """
    Class to compute the topological matrix of a given parcellation.

    Assume L is the number of cortical areas in a parcellation.  
    The parcellation is converted to an LxL matrix, where entry (i,j) 
    is the number of voxels in area (j) sharing an edge with voxels in areas (j).

    Parameters:
    - - - - -
    max_labels: int
        maximum number of expected labels in parcellation
    """

    def __init__(self, max_label=180):

        self.max_label = max_label
    
    def fit(self, label, adj):

        """
        Parameters:
        - - - - -
        label: int, array
            vector of label assignments for each voxel
            assumes the labels start at 1, and end at ```max_label```
        adj: dict
            adjacency list of surface over which the parcellation is distributed
        """

        n = self.max_label + 1

        label_map = {k: None for k in range(1, n)}
        for i in label_map.keys():
            label_map[i] = np.where(label == i)[0]
            
        L = np.zeros((n, n))
        for k,v in label_map.items():

            # identify voxels adjacent to those with label ```k```
            if k in label:
                neighbors = [adj[i] for i in v]
                neighbors = np.unique(np.concatenate(neighbors))
                # get labels of adjacent voxels
                adj_labels = label[neighbors]
                # compute number of instances of each adjacent label
                [l, c] = np.unique(adj_labels, return_counts=True)

                L[k, l] = c
        
        L = L / L.sum(1)[:,None]
        L[np.isnan(L)] = 0
        
        self.tpm = L[1:,1:]