### SOME SIMPLE TOOLS FOR SAVING DATA DURING LARGE SEARCHES ###

import h5py
import numpy as np
import os
import io
from PIL import Image

# saves fig and calculated interaction coefficients to h5 file
def add_fig_to_h5(h5_file, fig, fig_name, coefficients):
    if f'{fig_name}_coefficients' in f:
        del f[f'{fig_name}_coefficients']

    h5_file.create_dataset(
        f'{fig_name}_coefficients',
        data = coefficients
    )
   
    # store figure image as pre-compressed byte-array
    buf = io.BytesIO()

    fig.savefig(buf, format='png', bbox_inches='tight')
    binary_data =  np.frombuffer(buf.getvalue(), dtype=np.uint8)

    if fig_name in f:
        del f[fig_name]

    # add a dataset for the image and the coefficients
    h5_file.create_dataset(
        fig_name,
        data=binary_data,
        compression='gzip',
        compression_opts=9
    )

# load figure and calculated interaction coefficients from h5 file
def load_fig_from_h5(h5_file, fig_name):
    coef_dataset = h5_file[f'{fig_name}_coefficients']
    coef_array = coef_dataset[:] # [c6d, c6e, c3d, c3d]

    # load the plot
    binary_data = h5_file[fig_name][()]
    image_stream = io.BytesIO(binary_data)
    fig = Image.open(image_stream)

    return coef_array, fig


