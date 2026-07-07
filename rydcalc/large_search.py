### CODE FOR PERFORMING LARGE SEARCH OVER DUAL SPECIES INTERACTIONS ###

from .data_saving import *
from .pre_computation import *
from .analysis import *

import h5py
from pathlib import Path
from multiprocess import Pool
from multiprocess import Lock

lock = Lock()

'''
tool for searching over a large range of rydberg state configurations of two different atomic species
'''

class rydberg_pair_search:
    def __init__(self, atom1, atom2, a1_n_range, a2_n_range, Bz_range, opts, save_file):
        # INPUTS
        # - atom1: the first atom in the dual species array
        # - atom2: the second atom in the dual species array
        # - a1_n_range: the range of principle quantum numbers of atom 1 over which to search
        #   - should be a tuple of the form (n0, nf)
        # - a2_n_range: the range of principle quantum numbers of atom 2 over which to search
        #   - should be a tuple of the form (n0, nf)
        # - Bz_range: the range of field values over which to search
        # - opts: the options to pass to analysis_pair_interaction
        # - save_file: name of h5 file to store results in

        self.atom1 = atom1
        self.atom2 = atom2
        self.a1_n_range = a1_n_range
        self.a2_n_range = a2_n_range
        self.Bz_range = Bz_range
        self.opts = opts
        self.save_file = save_file

        if Path(self.save_file).is_file():
            print('warning: save file aready exists. search may fail to overwrite previous data')

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

    def generate_search_space(self, num_atomic_configs, num_field_configs):
        # generates a search space with shape (num_atomic_configs, num_field_configs)
        # returns list of tuples consisting of all configurations to search over
        #   - [(atom1_state, atom2_state, Bz), ...]
        
        # sample uniformly from a1_states and a2_states with the same spacing
        num_a1_plus_num_a2 = np.sqrt(num_atomic_configs * (len(self.a1_states) + len(self.a2_states))**2 / (len(self.a1_states) * len(self.a2_states)))

        num_from_a1 = int(len(self.a1_states) * num_a1_plus_num_a2 / (len(self.a1_states) + len(self.a2_states)))
        num_from_a2 = int(len(self.a2_states) * num_a1_plus_num_a2 / (len(self.a1_states) + len(self.a2_states)))

        indices_a1 = np.linspace(0, len(self.a1_states) - 1, num=num_from_a1, dtype=int)
        indices_a2 = np.linspace(0, len(self.a2_states) - 1, num=num_from_a2, dtype=int)

        # sample uniformly from the field values
        field_values = np.linspace(self.Bz_range[0], self.Bz_range[-1]-1, num=num_field_configs, dtype=float)

        self.search_space = []

        for i in indices_a1:
            for j in indices_a2:
                for field_value in field_values:
                    already_included = False
                    for element in self.search_space:
                        # dont include repeats for same atom interactions
                        if (element[0] == self.a2_states[j] and element[1] == self.a1_states[i] and element[2] == field_value):
                            already_included = True
                            break
                    
                    if not already_included:
                        self.search_space.append((self.a1_states[i], self.a2_states[j], field_value))

        return self.search_space
    
    def run_single(self, configuration):
        # run rydcalc interaction analysis on a single pair of states and field
        s1, s2, Bz = configuration
        
        rList_um = np.arange(2, 10, 0.2)

        pb = pair_basis_pre_computation()
        pb.fill(pair(s1, s2), include_opts=self.opts)
        pb.computeHamiltonians(multipoles=[[1,1]])

        pair_int = analysis_pair_interaction(s1, s2, include_opts=self.opts)
        result = pair_int.run(rList_um=rList_um, th=0, Bz_Gauss=Bz)

        fig,ax = pair_int.pa_plot(include_plot_opts={'ov_norm': 'log', 'log_norm': [0.01, 1]})
        ax[-1].set_yscale('symlog')

        # save the results
        with lock:
            with h5py.File(self.save_file, 'a') as h5_file:
                add_fig_to_h5(h5_file, fig, str(configuration), result)
    
    def run_search(self):
        # run the search, calculating the 

        if self.search_space is None:
            print('Error, please define search space before running search')
            return
        
        # otherwise start the search, running different parts in parallel
        with Pool(processes=4) as pool:
            results = pool.map(self.run_single, self.search_space)
        



