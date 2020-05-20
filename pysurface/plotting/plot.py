import numpy as np
import nibabel as nb
from brainspace import mesh, plotting


def plot_surf_hemi(surface, labels, scalar_map, view=['lateral', 'medial'], 
                   screenshot=False, filename=None, 
                   embed_nb=False, interactive=False,
                   cmap='Spectral_r', color_bar='right', 
                   nan_color=(0.85, 0.85, 0.85, 1), 
                   zoom=1.5, size=(1000, 800), 
                   **kwargs):
    
    """
    Plot scalar map data for single hemisphere.
    
    Parameters:
    - - - - -
    surface: string
        path to surface file
    labels: list
        label for each scalar map
        e.g. ['Correlation', 'P-value', 'label']
    scalar_map: list
        maps to plot on surface
        must have same number of indices as there are vertices in the surfaces
    view: list
        angles to plot surface of
    screenshot: bool
        save PNG file of figure
    filename: string
        PNG file name
    cmap: string
        colormap to use
    color_bar: string
        position of colorbar
    nan_color: tuple
        RGBA color value, normalized to [0, 1]
    zoom: float
        scaling factor of figure
    size: tuple
        total figure size, in pixels
        
    To change the colorbar or label text, we can supply a dictionary as follows:

    {'cb__width': ,
     'cb__height': ,
     'text__textProperty': {'fontSize': '},
    }
    """
    
    # If we want to save the image to a file
    # must provide ```screenshot``` and ```filename``` variables
    if screenshot and not filename:
        raise ValueError("If saving figure, must provide a PNG file path to ```filename```.")
    
    surf = mesh.mesh_io.read_surface(surface)
    n_pts = surf.n_points
    
    surfs = {'s': surf}
    layout = ['s' for k in np.arange(len(view))]
    
    kwds = {'view': view, 'share': 'r'}
    kwds.update(kwargs)
    
    if isinstance(scalar_map, np.ndarray):
            if array_name.ndim == 2:
                array_name = [a for a in scalar_map]
            elif array_name.ndim == 1:
                array_name = [scalar_map]

    if isinstance(scalar_map, list):
        layout = [layout] * len(scalar_map)
        array_name2 = []
        for an in scalar_map:
            if isinstance(an, np.ndarray):
                name = surf.append_array(an[:n_pts], at='p')
                array_name2.append(name)
            else:
                array_name2.append(an)
                
        array_name = np.asarray(array_name2)[:, None]
    
    color_range=None

    F = plotting.plot_surf(surfs, layout, array_name=array_name, label_text=labels,
                                      color_bar=color_bar, cmap=cmap, zoom=zoom, size=size,
                                      interactive=interactive, embed_nb=embed_nb, screenshot=screenshot,
                                      filename=filename, **kwds, nan_color=nan_color)
    
    return F