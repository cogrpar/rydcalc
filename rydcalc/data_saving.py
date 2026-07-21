### SOME SIMPLE TOOLS FOR SAVING DATA DURING LARGE SEARCHES ###

import h5py
import numpy as np
import os
import io
from PIL import Image
import dill

# saves fig and calculated interaction coefficients to h5 file
def add_fig_to_h5(h5_file, fig, fig_name, coefficients, opts, energies=None):
    if f'{fig_name}_coefficients' in h5_file:
        del h5_file[f'{fig_name}_coefficients']

    h5_file.create_dataset(
        f'{fig_name}_coefficients',
        data = coefficients
    )

    opts_str = dill.dumps(opts)
    if f'{fig_name}_opts' in h5_file:
        del h5_file[f'{fig_name}_opts']

    h5_file.create_dataset(
        f'{fig_name}_opts',
        data = np.void(opts_str),
    )

    if energies:
        if f'{fig_name}_energies' in h5_file:
            del h5_file[f'{fig_name}_energies']

        h5_file.create_dataset(
            f'{fig_name}_energies',
            data = energies # allows for storing a list of relevant energies during a calculation
        )

    # store figure image as pre-compressed byte-array
    buf = io.BytesIO()

    fig.savefig(buf, format='png', bbox_inches='tight')
    binary_data =  np.frombuffer(buf.getvalue(), dtype=np.uint8)

    if fig_name in h5_file:
        del h5_file[fig_name]

    # add a dataset for the image and the coefficients
    h5_file.create_dataset(
        fig_name,
        data=binary_data,
        compression='gzip',
        compression_opts=9
    )

# load figure and calculated interaction coefficients from h5 file
def load_fig_from_h5(h5_file, fig_name, return_energies=False):
    coef_dataset = h5_file[f'{fig_name}_coefficients']
    coef_array = coef_dataset[:] # [c6d, c6e, c3d, c3d]

    opts_dataset = h5_file[f'{fig_name}_opts']
    opts = dill.loads((opts_dataset[()]).tobytes())

    # load the plot
    binary_data = h5_file[fig_name][()]
    image_stream = io.BytesIO(binary_data)
    fig = Image.open(image_stream)

    if return_energies and f'{fig_name}_energies' in h5_file:
        energies = h5_file['{fig_name}_energies'][:]
        return fig, coef_array, opts, energies

    return fig, coef_array, opts