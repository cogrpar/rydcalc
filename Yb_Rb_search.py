import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cs
import time

import rydcalc

# fix deprecated function calls for newer versions of numpy and scipy
rydcalc.large_search.fix_libs()

logs_folder = './search_logs'     # TODO: change this to be some location in /n/scratch
on_cluster = False                # TODO: change to true 
save_file = './initial_search.h5' # TODO: change this to be some location in /n/scratch/

includefn = lambda p,p0: True if ((abs(p0.energy_Hz-p.energy_Hz)<5e9 ) and ((-1)**(p.s1.channels[0].l + p.s2.channels[0].l)==(-1)**(p0.s1.channels[0].l + p0.s2.channels[0].l))) else False

opts_interspecies = {'dn': 5,'dl': 3,'dm': 3,'dipole_allowed': False, 'pair_include_fn': includefn}
opts_intraspecies = {'dn': 4,'dl': 3,'dm': 3,'dipole_allowed': False, 'pair_include_fn': includefn} # because Yb is slow

Yb171 = rydcalc.Ytterbium171(cpp_numerov=True,use_db=False)
Rb = rydcalc.Rubidium87(cpp_numerov=True,use_db=False)

search = rydcalc.rydberg_pair_search(
    atom1 = Yb171, 
    atom2 = Rb, 
    a1_n_range = [50, 70], # TODO, does this look like a good range?
    a2_n_range = [50, 70], # TODO, does this look like a good range?
    Bz_range = [10, 10],   # TODO, does this look like a good range?
    opts = opts_interspecies, 
    opts_intraspecies = opts_intraspecies,
    save_file = save_file,
    logs_folder = logs_folder,
    on_cluster = on_cluster
)

search.generate_search_space(num_atomic_configs = 150, num_field_configs = 1)
assert(len(search.search_space_a1a2) == len(search.search_space_a1a1) * len(search.search_space_a2a2))

if __name__ == '__main__':
    search.run_search()