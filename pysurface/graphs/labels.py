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
        label : path to label file.  Can be either label.gii or dlabel.nii
        surfAdj : surface adjacency list
        k : path to midline indices

    """

    def __init__(self, label, adj_list, mids):

        self.label = label
        self.surf_adj = adj_list
        self.mids = mids

    def generate_adjList(self, filter_indices=None):

        """

        Method to compute label adjacency list for a given subject.
        If provided, will save the label adjacency list to provided filename

        """

        labels = self.label
        mids = self.mids
        adj_list = self.adj_list

        # get unique non-midline labels in cortical map
        labs = set(labels).difference({0, -1})

        labAdj = {}.fromkeys(labs)

        # Loop over unique values in label map
        for k in labs:

            # Find vertices belonging to parcel with current label

            tempInds = np.where(labels == k)[0]

            nLabels = []

            # Loop over vertices in each parcel
            for j in tempInds:

                # get neighbors, remove vertices included in midline

                neighbors = list(set(adj_list[j]).difference(set(mids)))
                nLabels.append(labels[neighbors])

            exclude = set([k, 0, -1])
            include = set(np.concatenate(nLabels))
            cleaned = list(include.difference(exclude))
            labAdj[k] = cleaned

        self.adj_list = labAdj
