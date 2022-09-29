#!/usr/bin/env python
import os
import time
import numpy as np
import pandas as pd
from mpi4py import MPI
from pathlib import Path
from simsopt import make_optimizable
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition
from neat.fields import Simple
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Simple
mpi = MpiPartition()
this_path = Path(__file__).parent.resolve()
def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)
############################################################################
#### Input Parameters
############################################################################
QA_or_QH = 'QH'
opt_quasisymmetry = False
opt_well = False

s_initial = 0.3  # initial normalized toroidal magnetic flux (radial VMEC coordinate)
nparticles = 500  # number of particles
tfinal = 5e-4  # total time of tracing in seconds
nsamples = 3000 # number of time steps
multharm = 3 # angular grid factor
ns_s = 3 # spline order over s
ns_tp = 3 # spline order over theta and phi
nper = 300 # number of periods for initial field line
npoiper = 200 # number of points per period on this field line
npoiper2 = 120 # points per period for integrator step
notrace_passing = 0 # if 1 skips tracing of passing particles, else traces them

nruns_opt_average = 1 # number of particle tracing runs to average over in cost function

nruns_robustness = 10 # number of runs when testing for robustness of cost function
output_path_parameters=os.path.join(this_path, 'opt_parameters.csv') # where to save robustness results
mult_factor = 3 # factor to multiply each variable in the regression test

if QA_or_QH == 'QA':
    B_scale = 8.58 / 2
    Aminor_scale = 8.5 / 2
else:
    B_scale = 6.55 / 2
    Aminor_scale = 12.14 / 2

######################################
######################################
if QA_or_QH == 'QA': filename = os.path.join(os.path.dirname(__file__), 'initial_configs', 'input.nfp2_QA')
else: filename = os.path.join(os.path.dirname(__file__), 'initial_configs', 'input.nfp4_QH')
vmec = Vmec(filename, mpi=mpi, verbose=False)
vmec.keep_all_files = True
surf = vmec.boundary
g_particle = ChargedParticleEnsemble(r_initial=s_initial)
######################################
OUT_DIR=os.path.join(this_path,f'out_s{s_initial}_NFP{vmec.indata.nfp}')
if opt_quasisymmetry: OUT_DIR+=f'_{QA_or_QH}'
if opt_well: OUT_DIR+=f'_well'
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
######################################
def EPcostFunction(v: Vmec):
    v.run()
    g_field_temp = Simple(wout_filename=v.output_file, B_scale=B_scale, Aminor_scale=Aminor_scale, multharm=multharm,ns_s=ns_s,ns_tp=ns_tp)
    final_loss_fraction_array = []
    for i in range(nruns_opt_average):
        g_orbits_temp = ParticleEnsembleOrbit_Simple(g_particle,g_field_temp,tfinal=tfinal,nparticles=nparticles,nsamples=nsamples,notrace_passing=notrace_passing,nper=nper,npoiper=npoiper,npoiper2=npoiper2)
        final_loss_fraction_array.append(g_orbits_temp.total_particles_lost)
    final_loss_fraction = np.mean(final_loss_fraction_array)
    # if not regression_test: print(f'Loss fraction = {final_loss_fraction}')
    # print(f'VMEC dofs = {v.x}')
    return final_loss_fraction
optEP = make_optimizable(EPcostFunction, vmec)
######################################
initial = 0
variables_for_testing = ['initial','nparticles','tfinal','nper','npoiper','npoiper2','nsamples']#,'ns_tp','multharm','ns_s','notrace_passing']
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n_tests = len(variables_for_testing)
perrank = n_tests//size
remainder = n_tests % size
start_iterator_MPI = rank * perrank + min(rank, remainder)
stop_iterator_MPI = (rank + 1) * perrank + min(rank + 1, remainder)
comm.Barrier()
start_time_tests = time.time()
# print(f'Core {rank+1}/{size} is doing',*range(start_iterator_MPI, stop_iterator_MPI),'iterators')
for i in range(start_iterator_MPI, stop_iterator_MPI):
    variable = variables_for_testing[i]
    # Increase parameters
    if variable in ['ns_s','ns_tp','multharm']:
        if eval(f"{variable}")==3: exec(f'{variable}=5')
        elif eval(f"{variable}")==5: exec(f'{variable}=3')
        # elif eval(f"{variable}")==4: exec(f'{variable}=5')
        else: continue
    elif variable == 'notrace_passing':
        if notrace_passing == 0: notrace_passing = 1
        elif notrace_passing == 1: notrace_passing = 0
    else: exec(f'{variable}=eval("{variable}")*{mult_factor}')
    # Go
    print(f'Running set of tests {i+1}/{n_tests} in core {rank+1}/{size} by increasing {variable}')
    start_time = time.time()
    costfunction_array=[]
    for i in range(nruns_robustness):
        costfunction_array.append(optEP.J())
    # Write results
    total_time = time.time() - start_time
    params_dict = {
        'QA_or_QH': QA_or_QH,
        's_initial': s_initial,
        'nparticles': nparticles,
        'tfinal': tfinal,
        'nsamples': nsamples,
        'multharm': multharm,
        'ns_s': ns_s,
        'ns_tp': ns_tp,
        'nper': nper,
        'npoiper': npoiper,
        'npoiper2': npoiper2,
        'notrace_passing': notrace_passing,
        'nruns': nruns_robustness,
        'duration_per_run': total_time/nruns_robustness,
        'std_J': np.std(costfunction_array),
        'mean_J': np.mean(costfunction_array),
        'coeff_variation': np.std(costfunction_array)/np.mean(costfunction_array)
    }
    df = pd.DataFrame(data=[params_dict])
    if not os.path.exists(output_path_parameters): pd.DataFrame(columns=df.columns).to_csv(output_path_parameters, index=False)
    df.to_csv(output_path_parameters, mode='a', header=False, index=False)
    # Revert back variables
    if variable in ['ns_s','ns_tp','multharm']:
        if eval(f"{variable}")==5: exec(f'{variable}=3')
        elif eval(f"{variable}")==3: exec(f'{variable}=5')
        # elif eval(f"{variable}")==4: exec(f'{variable}=3')
    elif variable == 'notrace_passing':
        if notrace_passing == 0: notrace_passing = 1
        elif notrace_passing == 1: notrace_passing = 0
    else: exec(f'{variable}=eval("{variable}")/{mult_factor}')