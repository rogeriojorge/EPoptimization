#!/usr/bin/env python
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from matplotlib.animation import FuncAnimation
from pathlib import Path
this_path = Path(__file__).parent.resolve()
from simsopt import make_optimizable
from simsopt.mhd import Vmec
from neat.fields import Simple
from simsopt.mhd import QuasisymmetryRatioResidual
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Simple

min_bound = -0.25
max_bound = 0.25
vmec_index_scan_opt = 0
npoints_scan = 100
tfinal = 1e-4
nparticles = 1500
ftol = 1e-2

MAXITER = 10
MAXFUN = 50
MAXITER_LOCAL = 2
MAXFUN_LOCAL = 5
run_scan = False
run_optimization = True
plot_result = True

output_path_parameters_opt = 'opt_dofs_loss.csv'
output_path_parameters_scan = 'scan_dofs_loss.csv'
output_path_parameters_min = 'min_dofs_loss.csv'

os.makedirs(os.path.join(this_path,'test_optimization'), exist_ok=True)
os.chdir(os.path.join(this_path,'test_optimization'))
vmec = Vmec(os.path.join(this_path,'initial_configs','input.QI'), verbose=False)
qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)    
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
surf.fix("rc(0,0)")
output_to_csv = True

def output_dofs_to_csv(csv_path,dofs,mean_iota,aspect,loss_fraction,quasisymmetry,well,effective_1o_time=0):
    keys=np.concatenate([[f'x({i})' for i, dof in enumerate(dofs)],['mean_iota'],['aspect'],['loss_fraction'],['quasisymmetry'],['well'],['effective_1o_time']])
    values=np.concatenate([dofs,[mean_iota],[aspect],[loss_fraction],[quasisymmetry],[well],[effective_1o_time]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(csv_path): pd.DataFrame(columns=df.columns).to_csv(csv_path, index=False)
    df.to_csv(csv_path, mode='a', header=False, index=False)
def EPcostFunction(v:Vmec):
    start_time = time.time()
    try:
        v.run()
        g_particle = ChargedParticleEnsemble(r_initial=0.3)
        g_field = Simple(wout_filename=v.output_file, B_scale=5.7/v.wout.b0/2, Aminor_scale=1.7/v.wout.Aminor_p/2)
        g_orbits = ParticleEnsembleOrbit_Simple(g_particle,g_field,tfinal=tfinal,nparticles=nparticles,nsamples=1000,notrace_passing=1)
        loss_fraction = g_orbits.total_particles_lost
        lost_times_array = tfinal-g_field.params.times_lost
        lost_times_array = lost_times_array[lost_times_array!=0.0]
        if np.asarray(lost_times_array).size==0: lost_times_array=[tfinal]
        effective_1o_time = np.mean(1/lost_times_array)/(np.max(1/lost_times_array)+1e-9)
        print(f'Loss fraction {loss_fraction:1f} for point {(vmec.x[vmec_index_scan_opt]):1f}, aspect {np.abs(v.aspect()):1f} and iota {(v.mean_iota()):1f} took {(time.time()-start_time):1f}s')
        if output_to_csv: output_dofs_to_csv(output_path_parameters_opt, v.x,v.mean_iota(),v.aspect(),loss_fraction,qs.total(),v.vacuum_well(),effective_1o_time=effective_1o_time)
        else: output_dofs_to_csv(output_path_parameters_scan, v.x,v.mean_iota(),v.aspect(),loss_fraction,qs.total(),v.vacuum_well(),effective_1o_time=effective_1o_time)
    except Exception as e:
        print(e,'Return loss fraction of 100%.')
        loss_fraction = 1
        print(f'Loss fraction {loss_fraction:1f} for point {(vmec.x[vmec_index_scan_opt]):1f} took {(time.time()-start_time):1f}s')
    return loss_fraction
optEP = make_optimizable(EPcostFunction, vmec)
def fun(dofss):
    vmec.x = [dofss[0] if count==vmec_index_scan_opt else vx for count,vx in enumerate(vmec.x)]
    return optEP.J()
if run_optimization:
    if os.path.exists(output_path_parameters_opt): os.remove(output_path_parameters_opt)
    if os.path.exists(output_path_parameters_min): os.remove(output_path_parameters_min)
    output_to_csv = True
    bounds = [(min_bound,max_bound)]
    minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds, "options": {'maxiter': MAXITER_LOCAL, 'maxfev': MAXFUN_LOCAL, 'disp': True}}
    global_minima_found = []
    def print_fun(x, f, context):
        if context==0: context_string = 'Minimum detected in the annealing process.'
        elif context==1: context_string = 'Detection occurred in the local search process.'
        elif context==2: context_string = 'Detection done in the dual annealing process.'
        else: print(context)
        print(f'New minimum found! x={x[0]:1f}, f={f:1f}. {context_string}')
        output_dofs_to_csv(output_path_parameters_min,vmec.x,vmec.mean_iota(),vmec.aspect(),f,qs.total(),vmec.vacuum_well())
        if len(global_minima_found)>4 and np.abs((f-global_minima_found[-1])/f)<ftol:
            # Stop optimization
            return True
        else:
            global_minima_found.append(f)
    no_local_search = False
    res = dual_annealing(fun, bounds=bounds, maxiter=MAXITER, maxfun=MAXFUN, x0=[random.uniform(min_bound,max_bound)], no_local_search=no_local_search, minimizer_kwargs=minimizer_kwargs, callback=print_fun)
    print(f"Global minimum: x = {res.x}, f(x) = {res.fun}")
    vmec.x = [res.x[0] if count==vmec_index_scan_opt else vx for count,vx in enumerate(vmec.x)]
if run_scan:
    if os.path.exists(output_path_parameters_scan): os.remove(output_path_parameters_scan)
    output_to_csv = False
    for point1 in np.linspace(min_bound,max_bound,npoints_scan):
        vmec.x = [point1 if count==vmec_index_scan_opt else vx for count,vx in enumerate(vmec.x)]
        loss_fraction = optEP.J()
if plot_result:
    df_scan = pd.read_csv(output_path_parameters_scan)

    try:
        df_opt = pd.read_csv(output_path_parameters_opt)
        fig, ax = plt.subplots()
        plt.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['loss_fraction'], label='Scan')
        ln, = ax.plot([], [], 'ro', markersize=1)
        vl = ax.axvline(0, ls='-', color='r', lw=1)
        patches = [ln,vl]
        ax.set_xlim(min_bound,max_bound)
        ax.set_ylim(np.min(0.8*df_scan['loss_fraction']), np.max(df_scan['loss_fraction']))
        def update(frame):
            ind_of_frame = df_opt.index[df_opt[f'x({vmec_index_scan_opt})'] == frame][0]
            df_subset = df_opt.head(ind_of_frame+1)
            xdata = df_subset[f'x({vmec_index_scan_opt})']
            ydata = df_subset['loss_fraction']
            vl.set_xdata([frame,frame])
            ln.set_data(xdata, ydata)
            return patches
        ani = FuncAnimation(fig, update, frames=df_opt[f'x({vmec_index_scan_opt})'])
        ani.save('opt_animation.gif', writer='imagemagick', fps=5)

        fig = plt.figure()
        plt.plot(df_opt[f'x({vmec_index_scan_opt})'], df_opt['loss_fraction'], 'ro', markersize=1, label='Optimizer')
        plt.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['loss_fraction'], label='Scan')
        plt.ylabel('Loss fraction');plt.xlabel('RBC(1,0)');plt.legend();plt.savefig('loss_fraction_over_opt_scan.pdf')
    except Exception as e: print(e)
    points_scan = np.linspace(min_bound,max_bound,len(df_scan[f'x({vmec_index_scan_opt})']))
    fig = plt.figure();plt.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['loss_fraction'], label='Scan')
    plt.ylabel('Loss fraction');plt.xlabel('RBC(1,0)');plt.legend();plt.savefig('loss_fraction_scan.pdf')
    fig = plt.figure();plt.plot(points_scan, df_scan['aspect'], label='Aspect ratio')
    plt.ylabel('Aspect ratio');plt.xlabel('RBC(1,0)');plt.savefig('aspect_ratio_scan.pdf')
    fig = plt.figure();plt.plot(points_scan, df_scan['mean_iota'], label='Rotational Transform (1/q)')
    plt.ylabel('Rotational Transform (1/q)');plt.xlabel('RBC(1,0)');plt.savefig('iota_scan.pdf')
    fig = plt.figure();plt.plot(points_scan, df_scan['quasisymmetry'], label='Quasisymmetry cost function')
    plt.ylabel('Quasisymmetry cost function');plt.xlabel('RBC(1,0)');plt.savefig('quasisymmetry_scan.pdf')
    fig = plt.figure();plt.plot(points_scan, df_scan['well'], label='Magnetic well')
    plt.ylabel('Magnetic well');plt.xlabel('RBC(1,0)');plt.savefig('magnetic_well_scan.pdf')
    fig = plt.figure();plt.plot(points_scan, df_scan['effective_1o_time'], label='Effective 1/time')
    plt.ylabel('Effective time');plt.xlabel('RBC(1,0)');plt.savefig('effective_1o_time_scan.pdf')