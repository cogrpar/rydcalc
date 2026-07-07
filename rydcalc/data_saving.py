### SOME SIMPLE TOOLS FOR SAVING DATA DURING LARGE SEARCHES ###

import h5py
import numpy as np
import os
import io

def add_fig_to_h5(h5_file, fig, fig_name, coefficients):
    h5_file.create_dataset(
        f'{fig_name}_coefficients',
        data = coefficients
    )
   
    # store figure image as pre-compressed byte-array
    buf = io.BytesIO()

    fig.savefig(buf, format='png', bbox_inches='tight')
    binary_data =  np.frombuffer(buf.getvalue(), dtype=np.uint8)

    # add a dataset for the image and the coefficients
    h5_file.create_dataset(
        fig_name,
        data=binary_data,
        compression="gzip",
        compression_opts=9
    )