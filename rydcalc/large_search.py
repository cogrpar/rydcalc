### CODE FOR PERFORMING LARGE SEARCH OVER DUAL SPECIES INTERACTIONS ###

from .data_saving import *
from .pre_computation import *
from .analysis import *

import h5py
from pathlib import Path
from collections import deque
from multiprocess import Pool
from filelock import FileLock
import math
import os
import sys
import shutil
import unicodedata
import subprocess
import time

'''
tool for searching over a large range of rydberg state configurations of two different atomic species
'''

# this must be called by all new processes that are spawned to ensure backwards compatibility with libraries
def fix_libs():
    if not hasattr(np, "product"):
        np.product = np.prod
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    np.math = math

    if not hasattr(scipy.special, 'sph_harm') and hasattr(scipy.special, 'sph_harm_y'):
        def _patched_sph_harm(m, n, theta, phi, out=None):
            '''Monkey patch to restore the legacy sph_harm signature safely.'''
            # if 'out' is explicitly passed as an array, we handle it manually
            if out is not None:
                # Compute result and copy it in-place into the provided 'out' array
                result = scipy.special.sph_harm_y(n, m, theta, phi)
                out[...] = result
                return out

            # standard case: sph_harm_y does not accept 'out' as a keyword argument
            return scipy.special.sph_harm_y(n, m, theta, phi)

        # bind the fix back to the module
        scipy.special.sph_harm = _patched_sph_harm

# for use with slurm jobs
def slurm_time_to_seconds(t):
    '''
    Convert SLURM time format to seconds.
    Handles:
      MM:SS
      HH:MM:SS
      D-HH:MM:SS
    '''
    days = 0

    if '-' in t:
        days, t = t.split('-')
        days = int(days)

    parts = list(map(int, t.split(':')))

    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f'Unexpected SLURM time format: {t}')

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def get_remaining_time():
    result = subprocess.run(
        ['squeue', '-h', '-j', os.environ['SLURM_JOB_ID'], '-o', '%L'],
        capture_output=True,
        text=True,
        check=True,
    )

    remaining = result.stdout.strip()

    if not remaining:
        return 0

    return slurm_time_to_seconds(remaining)

class rydberg_pair_search:
    def __init__(self, opts, atom1, atom2, a1_n_range, a2_n_range, Bz_range, Ez_range=[0, 0], theta_range=[0, 0], opts_intraspecies=None, save_file='search_results.h5', logs_folder='search_logs', on_cluster=False):
        # INPUTS
        # - opts: the options to pass to analysis_pair_interaction
        # - atom1: the first atom in the dual species array
        # - atom2: the second atom in the dual species array
        # - a1_n_range: the range of principle quantum numbers of atom 1 over which to search
        #   - should be a tuple of the form (n0, nf)
        # - a2_n_range: the range of principle quantum numbers of atom 2 over which to search
        #   - should be a tuple of the form (n0, nf)
        # - Bz_range: the range of B field values over which to search
        # - Ez_range: the range of E field values over which to search
        # - theta_range: the range of theta values over which to search
        # - save_file: name of h5 file to store results in

        self.atom1 = atom1
        self.atom2 = atom2
        self.a1_n_range = a1_n_range
        self.a2_n_range = a2_n_range
        self.Bz_range = Bz_range
        self.Ez_range = Ez_range
        self.theta_range = theta_range
        self.opts = opts
        self.opts_intraspecies = opts_intraspecies
        self.save_file = save_file

        self.on_cluster = on_cluster
        if on_cluster:
            self.allocated_cpus = min(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)), 25)
        else:
            self.allocated_cpus = None # let multiprocess use the number of available cores

        self.dual_species = (self.atom1 != self.atom2)

        # generate a full list of possible states within the provided range for atom 1
        self.a1_states = []

        qns = [self.a1_n_range[0], 0, 1/2, -1/2]
        nf = self.a1_n_range[1]

        state = self.atom1.get_state(tuple(qns))

        while qns[0] <= nf:
            self.a1_states.append(state)

            # go to the next state
            qns[3] += 1
            state = self.atom1.get_state(tuple(qns))
            if state:
                continue

            # if that didn't work, try to increase l or f depending on if this is an alkali or alkaline atom
            qns[2] += 1
            qns[3] = -qns[2]

            state = self.atom1.get_state(tuple(qns))
            if state:
                continue

            # if that didn't work, increase the principle quantum number
            if self.atom1.name == '171Yb': # quantum numbers v,l,f,m
                qns[0] += 1
                qns[2] = 1/2
                qns[3] = -qns[2]

            else: # quantum numbers n,l,j,m
                qns[0] += 1
                qns[2] = 1/2
                qns[3] = -qns[2]

            state = self.atom1.get_state(tuple(qns))

        # generate a full list of possible states within the provided range for atom 2
        self.a2_states = []

        qns = [self.a1_n_range[0], 0, 1/2, -1/2]
        nf = self.a1_n_range[1]

        state = self.atom2.get_state(tuple(qns))

        while qns[0] <= nf:
            self.a2_states.append(state)

            # go to the next state
            qns[3] += 1
            state = self.atom2.get_state(tuple(qns))
            if state:
                continue

            # if that didn't work, try to increase l or f depending on if this is an alkali or alkaline atom
            qns[2] += 1
            qns[3] = -qns[2]

            state = self.atom2.get_state(tuple(qns))
            if state:
                continue

            # if that didn't work, increase the principle quantum number
            if self.atom2.name == '171Yb': # quantum numbers v,l,f,m
                qns[0] += 1
                qns[2] = 1/2
                qns[3] = -qns[2]

            else: # quantum numbers n,l,j,m
                qns[0] += 1
                qns[2] = 1/2
                qns[3] = -qns[2]

            state = self.atom2.get_state(tuple(qns))


        # create the logs folder if it doesn't exist
        os.makedirs(logs_folder, exist_ok=True)
        self.logs_folder = logs_folder

        # get a list of completed calculations by looking through the logs
        self.completed_calculations = []

        for file_path in Path(logs_folder).iterdir():
            # check if it is a file
            if file_path.is_file():
                with open(file_path, 'r', encoding='utf-8') as f:
                    # keep only the last line in memory
                    last_line_list = deque(f, maxlen=1)

                    # convert the line to a string
                    if last_line_list:
                        last_line = last_line_list[0].strip()
                        if last_line[0] == '*':
                            self.completed_calculations.append(last_line_list[0].strip()[1:])

        #print(self.completed_calculations)

    def generate_search_space(self, num_atomic_configs, num_Bz_field_configs=1, num_Ez_field_configs=1, num_theta_configs=1, partitions = 1):
        # generates a search space with size as close to num_atomic_configs * num_field_configs * num_theta_configs as possible
        # generates 3 lists of tuples consisting of all configurations to search over
        #   - atom1-atom2 interactions: [(atom1_state, atom2_state, Bz, Ez, theta), ...]
        #   - atom1-atom1 interactions: [(atom1_state, atom1_state, Bz, Ez, theta), ...]
        #   - atom2-atom2 interactions: [(atom2_state, atom2_state, Bz, Ez, theta), ...]
        # sample uniformly from a1_states and a2_states with the same spacing

        s1 = len(self.a1_states)
        s2 = len(self.a2_states)

        s_ratio = s1/s2
        n2 = (-(s_ratio+1)+np.sqrt((s_ratio+1)**2+4*num_atomic_configs*s_ratio))/(2*s_ratio)
        n1 = s_ratio*n2

        num_from_a1 = int(np.round(n1))
        num_from_a2 = int(np.round(n2))

        print(f'atomic search space generated with size {num_from_a1*num_from_a2 + num_from_a1 + num_from_a2}')

        actual_atomic_configs = num_from_a1 + num_from_a2 + (num_from_a1*num_from_a2)

        indices_a1 = np.linspace(0, len(self.a1_states) - 1, num=num_from_a1, dtype=int)
        indices_a2 = np.linspace(0, len(self.a2_states) - 1, num=num_from_a2, dtype=int)

        # sample uniformly from the field values
        Bz_field_values = np.linspace(self.Bz_range[0], self.Bz_range[-1]-1, num=num_Bz_field_configs, dtype=float)
        Ez_field_values = np.linspace(self.Ez_range[0], self.Ez_range[-1]-1, num=num_Ez_field_configs, dtype=float)
        theta_values = np.linspace(self.theta_range[0], self.theta_range[-1]-1, num=num_theta_configs, dtype=float)

        search_space_a1a2 = []
        search_space_a1a1 = set()
        search_space_a2a2 = set()

        blank_counter = 0 # for keeping track of blank entries added to the search space to account for already calculated interactions
        for i in indices_a1:
            for j in indices_a2:
                for Bz_field_value in Bz_field_values:
                    for Ez_field_value in Ez_field_values:
                        for theta_value in theta_values:

                            config = (self.a1_states[i], self.a2_states[j], Bz_field_value, Ez_field_value, theta_value)
                            if str(config) in self.completed_calculations:
                                search_space_a1a2.append(f'blank{config}')
                                blank_counter += 1
                            else:
                                # add atom1 atom2 interaction to search space
                                search_space_a1a2.append(config)

                            config11 = (self.a1_states[i], self.a1_states[i], Bz_field_value, Ez_field_value, theta_value)
                            config22 = (self.a2_states[j], self.a2_states[j], Bz_field_value, Ez_field_value, theta_value)
                            # add atom1 and atom 2 intraspecies interaction to search space
                            if not str(config11) in self.completed_calculations:
                                search_space_a1a1.add(config11)
                            elif not f'blank{config11}' in search_space_a1a1:
                                search_space_a1a1.add(f'blank{config11}')
                                blank_counter += 1

                            if not str(config22) in self.completed_calculations:
                                search_space_a2a2.add(config22)
                            elif not f'blank{config22}' in search_space_a2a2:
                                search_space_a2a2.add(f'blank{config22}')
                                blank_counter += 1

        print(f'there are {blank_counter} entries in search space which have already been computed!')

        # partition the search space into [partitions] chunks
        self.search_space_a1a2 = np.array_split(np.array(search_space_a1a2, dtype=object), partitions)
        self.search_space_a1a1 = np.array_split(np.array(list(search_space_a1a1), dtype=object), partitions)
        self.search_space_a2a2 = np.array_split(np.array(list(search_space_a2a2), dtype=object), partitions)

    def run_single(self, configuration, save_file):
        fix_libs()

        # redirect stdoutput to log file
        original_stdout = sys.stdout

        log_filename = unicodedata.normalize('NFKD', str(configuration)).encode('ascii', 'ignore').decode('ascii')
        log_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', log_filename)
        log_filename = log_filename.strip(' ._-')

        with open(f'{self.logs_folder}/{log_filename}.log', 'w') as f:
            sys.stdout = f

            # run rydcalc interaction analysis on a single pair of states and field
            s1, s2, Bz, Ez, theta = configuration

            rList_um = np.arange(2, 10, 0.2)

            # get the correct opts if this is an intraspecies calculation
            if s1.atom.name == s2.atom.name and self.opts_intraspecies:
                opts = self.opts_intraspecies
            else:
                opts = self.opts

            start = time.perf_counter()

            print("entered", flush=True)

            print("before pb", flush=True)
            pb = pair_basis_pre_computation()

            print("before fill", flush=True)
            pb.fill(pair(s1, s2), include_opts=opts)

            print("before computeHamiltonians", flush=True)
            pb.computeHamiltonians(multipoles=[[1,1]])
            end = time.perf_counter()

            print(f'Computed Hamiltonian in {end-start} s')

            start = time.perf_counter()
            pair_int = analysis_pair_interaction(s1, s2, pb=pb, include_opts=opts)
            result = pair_int.run(rList_um=rList_um, th=theta, Bz_Gauss=Bz, Ez_Vcm=Ez)

            fig,ax = pair_int.pa_plot(include_plot_opts={'ov_norm': 'log', 'log_norm': [0.01, 1]}, sample_r=[2, 4, 6, 8])
            print(pair_int.sampled_energies)
            end = time.perf_counter()

            print(f'Computed interactions in {end-start} s')

            # save the results
            start = time.perf_counter()
            with FileLock(f'{save_file}.lock'):
                with h5py.File(save_file, 'a') as h5_file:
                    add_fig_to_h5(h5_file, fig, str(configuration), result, opts, energies=pair_int.sampled_energies)
            end = time.perf_counter()

            print(f'Saved results in {end-start} s')

            # the last line in the log file should be the configuration to signal successful completion
            print(f'*{configuration}')

        sys.stdout = original_stdout

    def run_search(self, on_partition=0, channels=['a1a1', 'a1a2', 'a2a2']):
        # run the search over entire search space

        if self.search_space_a1a2 is None:
            print('Error, please define search space before running search')
            return

        # otherwise start the search, running different parts in parallel
        print('starting search...')

        with Pool(processes=self.allocated_cpus, maxtasksperchild=5) as pool:
            if self.dual_species:
                sf12 = self.save_file.replace('.h5', '_interspecies.h5')
                sf11 = self.save_file.replace('.h5', '_intraspecies_a1.h5')
                sf22 = self.save_file.replace('.h5', '_intraspecies_a2.h5')
                search_args = [(tuple(config), sf12) for config in self.search_space_a1a2[on_partition] if not(type(config) is str) and 'a1a2' in channels] + \
                            [(tuple(config), sf11) for config in self.search_space_a1a1[on_partition] if not(type(config) is str) and 'a1a1' in channels] + \
                            [(tuple(config), sf22) for config in self.search_space_a2a2[on_partition] if not(type(config) is str) and 'a2a2' in channels]
                result = pool.starmap_async(self.run_single, search_args)

            else:
                search_args = [(tuple(config), self.save_file) for config in self.search_space_a1a2[on_partition] if not(type(config) is str)]
                result = pool.starmap_async(self.run_single, search_args)

            print('started pool...')
            while not result.ready():
                remaining = get_remaining_time() if self.on_cluster else 1e10

                if remaining < 300:
                    print('Less than 5 minutes remaining. Terminating pool.')
                    pool.terminate()
                    pool.join()
                    break

                time.sleep(30)