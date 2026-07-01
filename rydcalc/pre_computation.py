### SOME SIMPLE TOOLS FOR PRE COMPUTING MULTIPOLE MATRIX ELEMENTS ###
### FOR LATER USE IN COMPUTING THE HAMILTONIAN                    ###

import json
import ast
import time

from diskcache import Cache
from multiprocessing import Process
from multiprocessing import shared_memory

from .single_basis import *
from .pair_basis import *

import multiprocessing as mp
mp.set_start_method("fork")



'''
One option is to access all calculated dipole matrix elements directly in a hash table in
memory to speed up access time. This should be viable on the cluster with a table 
size on the order of 10 GB, and lookups will be faster than accessing them from a 
database using the built in (though not fully implemented) db_manager.
'''

def precompute_multipole_me(atom, qns : list, opts, num_center_states):
    # this function computes a wide range of multipole matrix elements for the passed atom
    # these values are saved so they can be reused when calculating interaction hamiltonians
    # INPUTS:
    #   - atom: atom for which the matrix elements are calculated
    #   - qns: the quantum numbers of the lowest energy state in the search
    #       - the search is performed over bases centered at states with subupquent quantum numbers
    #   - opts: options for computing basis for each center state
    #   - num_center_states: the number of center states to consider, steps through v and m_j

    mes = dict()
    state = atom.get_state(tuple(qns))

    for center_state_i in range(num_center_states):
        # compute the single state basis centered at the center state
        basis = single_basis()
        basis.fill(state, include_opts=opts)
        dim = basis.dim()

        # compute all dipole matrix elements
        for i in range(dim):
            for j in range(dim):
                key = (basis.states[i].__hash__(), basis.states[j].__hash__()) # states are hashable

                if not (key in mes.keys()):
                    me = basis.states[i].get_multipole_me(basis.states[j],k=1) # just dipole for now
                    mes[key] = me

                print(f'\rprogress: calculating center state {center_state_i+1}/{num_center_states}, i={i+1}/{dim}, j={j+1}/{dim}...', end='    ')

        # go to the next state
        qns[3] += 1
        state = atom.get_state(tuple(qns))
        if state:
            continue

        # if that didn't work, try to increase l or f depending on if this is an alkali or alkaline atom
        qns[2] += 1
        qns[3] = -qns[2]

        state = atom.get_state(tuple(qns))
        if state:
            continue
        
        # if that didn't work, increase the principle quantum number
        if atom.name == '171Yb': # quantum numbers v,l,f,m
            qns[0] += 1
            qns[2] = 1/2
            qns[3] = -qns[2]

        else: # quantum numbers n,l,j,m
            qns[0] += 1
            qns[2] = 1/2
            qns[3] = -qns[2]

        state = atom.get_state(tuple(qns))
        

    print()

    return mes

def save_mes_json(filename, mes):
    # write tuple keys to strings
    string_keyed_mes = {str((key[0], key[1])): value for key, value in mes.items()}

    # write to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(string_keyed_mes, f, indent=4)

    return 0

def load_mes_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        string_keyed_mes = json.load(file)
        mes = {ast.literal_eval(key): value for key, value in string_keyed_mes.items()}
        return mes
    


'''
Another (better) option is to use diskcache while calculating only the matrix
elements that will be explicitly needed in computing the Hamiltonian
'''

cache = Cache('multipole_matrix_elements')

class pair_basis_pre_computation(pair_basis):
    def __init__(self):
        super().__init__()

    @staticmethod
    def cached_multipole_me(initial_state, final_state, k=1, qIn=None, operator=None, pre_computed_mes=None):
        # wrapper for state.get_multipole_me that caches the result using diskcache
        
        key = ('multipole_me', str(initial_state), str(final_state), k, qIn)

        if key in cache:
            return cache[key]
        
        print('Cache miss!!')
        me = initial_state.get_multipole_me(final_state, k=k, qIn=qIn, operator=operator, pre_computed_mes=pre_computed_mes)
        cache[key] = me
        return me
    
    @staticmethod
    def cached_magnetic_me(initial_state, final_state):
        # wraper for state.atom.get_magnetic_me that caches the result using diskcache

        key = ('magnetic_me', str(initial_state), str(final_state))

        if key in cache:
            return cache[key]
        
        print('Cache miss!!')
        me = initial_state.atom.get_magnetic_me(initial_state, final_state)
        cache[key] = me
        return me
    
    @staticmethod
    def cached_magnetic_int(final_state, initial_state):
        # wraper for state.atom.diamagnetic_int that caches the result using diskcache

        key = ('diamagnetic_int', str(final_state), str(initial_state))

        if key in cache:
            return cache[key]
        
        print('Cache miss!!')
        me = initial_state.atom.diamagnetic_int(final_state, initial_state)
        cache[key] = me
        return me

    def _compute_HEz_Hint_fast(self, a1_precomputed_me=None, a2_precomputed_me=None):
        # override to utilize cached_multipole_me
        # must be called from computeHamiltonians or certain class attributes will not be instantiated
        
        printDebug = False
        computeHInt = True
        
        # generate a list of all unique states s1
        unique_s1 = list(set([p.s1 for p in self.pairs]))
        
        # for each state, make a list of the pair indexes where they occur
        unique_s1_pair_idx = []

        ttot=0

        for us1 in unique_s1:
            indices = [en[0] for en in filter(lambda p: p[1].s1==us1, enumerate(self.pairs))]
            unique_s1_pair_idx.append(indices)


        # now we loop over blocks of pairs with same s1
        for s1_i,pair_idx_i in zip(unique_s1,unique_s1_pair_idx):
            
            for s1_j,pair_idx_j in zip(unique_s1,unique_s1_pair_idx):

                # Ez
                # compute Ez if s1_i == s1_j or if there is a ME for s1_i->j
                if s1_i == s1_j or s1_i.allowed_multipole(s1_j,k=1,qIn=(0,)):
                    
                    s1_same = True if s1_i == s1_j else False

                    # loop over upper diagonal
                    
                    for ii in pair_idx_i:
                        for jj in filter(lambda x: x>= ii, pair_idx_j):
                            
                            pii = self.pairs[ii]
                            pjj = self.pairs[jj]
                            
                            if (not s1_same) and pii.s2 == pjj.s2:
                                self.HEz[ii,jj] += pair_basis_pre_computation.cached_multipole_me(pii.s1, pjj.s1, qIn=(0,), k=1)
                             
                            if s1_same and pii.s2.allowed_multipole(pjj.s2,k=1,qIn=(0,)):
                                self.HEz[ii,jj] += pair_basis_pre_computation.cached_multipole_me(pii.s2, pjj.s2, qIn=(0,), k=1)


                # now look at interactions

                for mm,HInt in zip(self.multipoles, self.HInt):
                    
                    # if there is the right transition allowed on s1
                    
                    if s1_i.allowed_multipole(s1_j,k=mm[0]):
                        
                        for ii in pair_idx_i:
                            for jj in filter(lambda x: x>= ii, pair_idx_j):
                                
                                pii = self.pairs[ii]
                                pjj = self.pairs[jj]
                                
                                # already checked s1
                                #if pii.s1.allowed_multipole(pjj.s1,k=mm[0]) and pii.s2.allowed_multipole(pjj.s2,k=mm[1]):
                                if pii.s2.allowed_multipole(pjj.s2,k=mm[1]):
                                    
                                    # q = final-initial, ie ii-jj
                                    q1 = -(pii.s1.m - pjj.s1.m)
                                    q2 = -(pii.s2.m - pjj.s2.m)
                
                                    qtot = int(q1+q2)
                                    qidx = qtot + (mm[0] + mm[1]) #index for HInt arr.

                                    d1 = pair_basis_pre_computation.cached_multipole_me(pii.s1, pjj.s1, k=mm[0], pre_computed_mes=a1_precomputed_me)
                                    d2 = pair_basis_pre_computation.cached_multipole_me(pii.s2, pjj.s2, k=mm[1], pre_computed_mes=a2_precomputed_me)

                                    #cg = CG(mm[0],q1,mm[1],q2,mm[0]+mm[1],q1+q2).doit().evalf()
                                    cg = CG(mm[0],q1,mm[1],q2,mm[0]+mm[1],q1+q2)
                
                                    me = cg*d1*d2
                
                                    if me != 0:
                                    # will multiply later by other factors which are not state-dependent
                                        HInt[qidx,ii,jj] = me
                
                                    if me != 0  and printDebug:
                                        print(pii, "<-(",qtot,mm,")-> ", pjj)

        # since we only did upper-right triangle, add transposed version
        self.HEz = self.HEz + np.conjugate(np.transpose(self.HEz))
        
        for mm,HInt in zip(self.multipoles, self.HInt): 
            for qidx in range(len(HInt)):
                HInt[qidx] = HInt[qidx] + np.conjugate(np.transpose(HInt[qidx])) - np.diagflat(np.diagonal(HInt[qidx]))

    def _compute_HBdiam(self, shm_name=None, shared_H_shape=None, shared_H_dtype=None):
        # override to utilize cached_multipole_me
        # must be called from computeHamiltonians or certain class attributes will not be instantiated

        # optionally use shared memory so the func can be called on a separate process

        existing_shm = None
        HBdiam = None

        if shm_name:
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            HBdiam = np.ndarray(shared_H_shape, dtype=shared_H_dtype, buffer=existing_shm.buf)

        # if we do, put in off-diagonal matrix elements as well
        # NB: this might fail for interactions between MQDT and non-MQDT, because we don't have get_magnetic_me defined for non-MQDT states.
        for ii in range(self.dim()):
            for jj in range(self.dim()):

                pii = self.pairs[ii]
                pjj = self.pairs[jj]

                if pii.s2 == pjj.s2:
                    HBdiam[ii, jj] += pair_basis_pre_computation.cached_magnetic_int(self.pairs[ii].s1, self.pairs[jj].s1)

                if pii.s1 == pjj.s1:
                    HBdiam[ii, jj] += pair_basis_pre_computation.cached_magnetic_int(self.pairs[ii].s2, self.pairs[jj].s2)

        if existing_shm:
            existing_shm.close()
        return
    
    def _compute_HBz(self, shm_name=None, shared_H_shape=None, shared_H_dtype=None):
        # override to optionally use shared memory so the func can be called on a separate process
        existing_shm = None
        HBz = None

        if shm_name:
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            HBz = np.ndarray(shared_H_shape, dtype=shared_H_dtype, buffer=existing_shm.buf)

        
        # if we don't have an MQDT model, can just enter diagonal matrix elements
        if (not isinstance(self.pairs[0].s1,state_mqdt)) or (not isinstance(self.pairs[0].s2,state_mqdt)):
            for ii in range(self.dim()):
                HBz[ii,ii] = self.pairs[ii].s1.get_g()*self.pairs[ii].s1.m + self.pairs[ii].s2.get_g()*self.pairs[ii].s2.m
                
            if existing_shm:
                existing_shm.close()
            return
    
        # if we do, put in off-diagonal matrix elements as well
        # NB: this might fail for interactions between MQDT and non-MQDT, because we don't have get_magnetic_me defined for non-MQDT states.
        for ii in range(self.dim()):
            for jj in range(self.dim()):
                
                pii = self.pairs[ii]
                pjj = self.pairs[jj]
                
                if pii.s2 == pjj.s2:
                    HBz[ii,jj] += pii.s1.atom.get_magnetic_me(self.pairs[ii].s1,self.pairs[jj].s1)
                    
                if pii.s1 == pjj.s1:
                    HBz[ii,jj] += pii.s2.atom.get_magnetic_me(self.pairs[ii].s2,self.pairs[jj].s2)
        
        if existing_shm:
            existing_shm.close()

        return

    def computeHamiltonians(self, multipoles=[[1,1]], a1_precomputed_me=None, a2_precomputed_me=None):
        # override to utilize ...
        
        self.multipoles = multipoles

        Nb = self.dim()

        self.HEz = np.zeros((Nb,Nb))
        self.HBz = np.zeros((Nb,Nb))
        self.HBdiam = np.zeros((Nb,Nb))
        self.H0 = np.zeros((Nb,Nb))

        sh_mem_HBz = shared_memory.SharedMemory(create=True, size=self.HBz.nbytes)
        shared_HBz = np.ndarray(self.HBz.shape, dtype=self.HBz.dtype, buffer=sh_mem_HBz.buf)
        np.copyto(shared_HBz, self.HBz)

        sh_mem_HBdiam = shared_memory.SharedMemory(create=True, size=self.HBdiam.nbytes)
        shared_HBdiam = np.ndarray(self.HBdiam.shape, dtype=self.HBdiam.dtype, buffer=sh_mem_HBdiam.buf)
        np.copyto(shared_HBdiam, self.HBdiam)
        
        # HInt is 5xNxN, where the first index is the total change
        # in magnetic quantum number. Keeping track of it this way allows
        # the angular dependence to be easily worked out later
        #self.HInt = np.zeros((5,Nb,Nb))
        
        self.HInt = []
        
        for mm in self.multipoles:
            # need to have an Nb x Nb matrix for each possible deltaMtotal
            dm_max = mm[0] + mm[1]
            self.HInt.append(np.zeros((2*dm_max+1,Nb,Nb)))
            

        # first, compue H0 and HBz, which are diagonal
        for ii in range(Nb):
            self.H0[ii,ii] = self.pairs[ii].energy_Hz - self.p0.energy_Hz
            #self.HBz[ii,ii] = self.pairs[ii].s1.get_g()*self.pairs[ii].s1.m + self.pairs[ii].s2.get_g()*self.pairs[ii].s2.m

        p_HBz = Process(target=self._compute_HBz, args=(sh_mem_HBz.name, self.HBz.shape, self.HBz.dtype))
        p_HBz.start()

        p_HBdiam = Process(target=self._compute_HBdiam, args=(sh_mem_HBdiam.name, self.HBdiam.shape, self.HBdiam.dtype))
        p_HBdiam.start()

        self._compute_HEz_Hint_fast(a1_precomputed_me=a1_precomputed_me, a2_precomputed_me=a2_precomputed_me)

        p_HBz.join()
        np.copyto(self.HBz, shared_HBz)

        sh_mem_HBz.close()
        sh_mem_HBz.unlink()

        # this usually takes longer to finish do don't bother joining until after cleaning up p_HBz
        p_HBdiam.join()
        np.copyto(self.HBdiam, shared_HBdiam)

        sh_mem_HBdiam.close()
        sh_mem_HBdiam.unlink()

        return
        