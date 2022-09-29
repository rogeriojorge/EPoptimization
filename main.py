#!/usr/bin/env python
import os
import glob
import time
import shutil
import vmecPlot2
import numpy as np
import pandas as pd
from mpi4py import MPI
import booz_xform as bx
from pathlib import Path
import matplotlib.pyplot as plt
from simsopt import make_optimizable
from simsopt.mhd import Vmec, Boozer
from simsopt.util import MpiPartition
from simsopt.solve import least_squares_mpi_solve, least_squares_serial_solve
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from neat.fields import Simple
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Simple
from scipy.optimize import minimize, basinhopping, differential_evolution, dual_annealing
from Alan_objectives import MaxElongationPen, MirrorRatioPen
mpi = MpiPartition()
this_path = Path(__file__).parent.resolve()
def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)
############################################################################
#### Input Parameters
############################################################################
MAXITER = 500
max_modes = [1, 2]
QA_or_QH_or_QI = 'QA'
opt_quasisymmetry = False
opt_EP = True
opt_well = False
opt_iota = False
opt_Mirror = True
opt_Elongation = True
plot_result = True
optimizer = 'dual_annealing' # least_squares_diff, least_squares, basinhopping, differential_evolution, dual_annealing
use_previous_results_if_available = False

weight_optEP = 100.0
weight_opt_Mirror = 100.0
weight_opt_Elongation = 10.0
weight_opt_well = 0.1
redux_B = 1.5 # Use ARIES-CS magnetic field reduced by this factor
redux_Aminor = 1.5 # Use ARIES-CS minor radius reduced by this factor
if QA_or_QH_or_QI == 'QA': aspect_ratio_target = 6
elif QA_or_QH_or_QI == 'QH': aspect_ratio_target = 7
elif QA_or_QH_or_QI == 'QI': aspect_ratio_target = 8

s_initial = 0.3  # initial normalized toroidal magnetic flux (radial VMEC coordinate)
nparticles = 600  # number of particles
tfinal = 6e-5  # total time of tracing in seconds
nsamples = 1500 # number of time steps
multharm = 3 # angular grid factor
ns_s = 3 # spline order over s
ns_tp = 3 # spline order over theta and phi
nper = 400 # number of periods for initial field line
npoiper = 150 # number of points per period on this field line
npoiper2 = 120 # points per period for integrator step
notrace_passing = 0 # if 1 skips tracing of passing particles, else traces them

nruns_opt_average = 1 # number of particle tracing runs to average over in cost function
iota_target = -0.42
well_target = 0.1

diff_rel_step = 1e-1
diff_abs_step = 1e-2

output_path_parameters=f'output_{optimizer}_{QA_or_QH_or_QI}.csv'
######################################
######################################
if QA_or_QH_or_QI == 'QA': nfp=2
elif QA_or_QH_or_QI == 'QH': nfp=4
elif QA_or_QH_or_QI == 'QI': nfp=3 # Change it later to vmec.indata.nfp
OUT_DIR_APPENDIX=f'out_s{s_initial}_NFP{nfp}'
if opt_quasisymmetry: OUT_DIR_APPENDIX+=f'_{QA_or_QH_or_QI}'
if opt_well: OUT_DIR_APPENDIX+=f'_well'
OUT_DIR = os.path.join(this_path, OUT_DIR_APPENDIX)
os.makedirs(OUT_DIR, exist_ok=True)
######################################
dest = os.path.join(OUT_DIR,OUT_DIR_APPENDIX+'_previous')
if use_previous_results_if_available and (os.path.isfile(os.path.join(OUT_DIR,'input.final')) or os.path.isfile(os.path.join(dest,'input.final'))):
    if MPI.COMM_WORLD.rank == 0:
        os.makedirs(dest, exist_ok=True)
        if os.path.isfile(os.path.join(OUT_DIR, 'input.final')) and not os.path.isfile(os.path.join(dest, 'input.final')):
            files = os.listdir(OUT_DIR)
            for f in files:
                shutil.move(os.path.join(OUT_DIR, f), dest)
    else:
        time.sleep(0.5)
    filename = os.path.join(dest, 'input.final')
else:
    if QA_or_QH_or_QI == 'QA': filename = os.path.join(this_path, 'initial_configs', 'input.nfp2_QA')
    elif QA_or_QH_or_QI == 'QH': filename = os.path.join(this_path, 'initial_configs', 'input.nfp4_QH')
    elif QA_or_QH_or_QI == 'QI': filename = os.path.join(this_path, 'initial_configs', 'input.QI')
os.chdir(OUT_DIR)
vmec = Vmec(filename, mpi=mpi, verbose=False)
vmec.keep_all_files = True
surf = vmec.boundary
g_particle = ChargedParticleEnsemble(r_initial=s_initial)
######################################
def output_dofs_to_csv(dofs,mean_iota,aspect,loss_fraction,eff_time,mirror_ratio,max_elongation):
    keys=np.concatenate([[f'x({i})' for i, dof in enumerate(dofs)],['mean_iota'],['aspect'],['loss_fraction'],['eff_time'],['mirror_ratio'],['max_elongation']])
    values=np.concatenate([dofs,[mean_iota],[aspect],[loss_fraction],[eff_time],[mirror_ratio],[max_elongation]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(output_path_parameters): pd.DataFrame(columns=df.columns).to_csv(output_path_parameters, index=False)
    df.to_csv(output_path_parameters, mode='a', header=False, index=False)
######################################
optElongation = make_optimizable(MaxElongationPen, vmec)
optMirror = make_optimizable(MirrorRatioPen, vmec)
######################################
def EPcostFunction(v: Vmec):
    start_time = time.time()
    try: v.run()
    except Exception as e:
        print(e)
        return 1e3
    B_scale = 5.7/v.wout.b0/redux_B  # Scale the magnetic field by a factor
    Aminor_scale = 1.7/v.wout.Aminor_p/redux_Aminor  # Scale the machine size by a factor
    g_field_temp = Simple(wout_filename=v.output_file, B_scale=B_scale, Aminor_scale=Aminor_scale, multharm=multharm,ns_s=ns_s,ns_tp=ns_tp)
    final_loss_fraction_array = []
    effective_time_array = []
    for i in range(nruns_opt_average): # Average over a given number of runs
        for j in range(0,3): # Try three times the same orbits, if not able continue
            while True:
                try:
                    g_orbits_temp = ParticleEnsembleOrbit_Simple(g_particle,g_field_temp,tfinal=tfinal,nparticles=nparticles,nsamples=nsamples,notrace_passing=notrace_passing,nper=nper,npoiper=npoiper,npoiper2=npoiper2)
                    final_loss_fraction_array.append(g_orbits_temp.total_particles_lost)
                    lost_times_array = tfinal-g_field_temp.params.times_lost
                    lost_times_array = lost_times_array[lost_times_array!=0.0]
                    if np.asarray(lost_times_array).size==0: lost_times_array=[tfinal]
                    effective_time_array.append(np.mean(lost_times_array)/(np.max(lost_times_array)+1e-9))
                except ValueError as error_print:
                    print(f'Try {j} of ParticleEnsembleOrbit_Simple gave error:',error_print)
                    continue
                break
    final_loss_fraction = np.mean(final_loss_fraction_array)
    final_effective_time = np.min([np.max([np.mean(effective_time_array),0]),10])
    g_field_temp.simple_main.finalize()
    mirror_ratio = MirrorRatioPen(v=v, output_mirror=True)
    max_elongation = MaxElongationPen(vmec=v, return_elongation=True)
    print(f'Loss = {(100*final_loss_fraction):1f}% with '
    # + f'eff time = {final_effective_time:1f} (J={(final_effective_time*final_loss_fraction):1f}) and '
    # + 'dofs = {v.x}, mean_iota={v.mean_iota()} and '
    + f'mirror ratio={mirror_ratio:1f}, max elongation={max_elongation:1f} and '
    + f'aspect ratio={v.aspect():1f} took {(time.time()-start_time):1f}s')
    output_dofs_to_csv(v.x,v.mean_iota(),v.aspect(),final_loss_fraction,final_effective_time,mirror_ratio,max_elongation)
    return final_loss_fraction
    # return final_effective_time*final_loss_fraction
optEP = make_optimizable(EPcostFunction, vmec)
######################################
try:
    pprint("Initial aspect ratio:", vmec.aspect())
    pprint("Initial mean iota:", vmec.mean_iota())
    pprint("Initial magnetic well:", vmec.vacuum_well())
    pprint("Initial mirror ratio:", MirrorRatioPen(v=vmec, output_mirror=True))
    pprint("Initial max elongation:", MaxElongationPen(vmec=vmec, return_elongation=True))
except Exception as e: pprint(e)
if MPI.COMM_WORLD.rank == 0:
    B_scale = 5.7/vmec.wout.b0/redux_B  # Scale the magnetic field by a factor
    Aminor_scale = 1.7/vmec.wout.Aminor_p/redux_Aminor  # Scale the machine size by a factor
    g_field = Simple(wout_filename=vmec.output_file, B_scale=B_scale, Aminor_scale=Aminor_scale, multharm=multharm,ns_s=ns_s,ns_tp=ns_tp)
    g_orbits = ParticleEnsembleOrbit_Simple(g_particle,g_field,tfinal=tfinal,nparticles=nparticles,notrace_passing=notrace_passing,nper=nper,npoiper=npoiper,npoiper2=npoiper2)
    pprint("Initial loss fraction:", g_orbits.total_particles_lost)
######################################
if QA_or_QH_or_QI == 'QA': qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
else: qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)    
opt_tuple = [(vmec.aspect, aspect_ratio_target, 1)]
if opt_well: opt_tuple.append((vmec.vacuum_well, well_target, weight_opt_well))
if opt_iota: opt_tuple.append((vmec.mean_iota, iota_target, 1))
if opt_EP: opt_tuple.append((optEP.J, 0, weight_optEP))
if opt_quasisymmetry: opt_tuple.append((qs.residuals, 0, 1))
if opt_Mirror: opt_tuple.append((optMirror.J, 0, weight_opt_Mirror))
if opt_Elongation: opt_tuple.append((optElongation.J, 0, weight_opt_Elongation))
try: pprint("Quasisymmetry objective before optimization:", qs.total())
except Exception as e: pprint(e)
######################################
initial_dofs=np.copy(surf.x)
def fun(dofss):
    # prob.x = initial_dofs
    prob.x = dofss
    return prob.objective()
for max_mode in max_modes:
    output_path_parameters=f'output_{optimizer}_{QA_or_QH_or_QI}_maxmode{max_mode}.csv'
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    initial_dofs=np.copy(surf.x)
    dofs=surf.x
    ######################################
    prob = LeastSquaresProblem.from_tuples(opt_tuple)
    if MPI.COMM_WORLD.rank == 0: pprint("Total objective before optimization:", prob.objective())
    pprint('-------------------------')
    pprint(f'Optimizing with max_mode = {max_mode}')
    pprint('-------------------------')
    if optimizer == 'minimize':
        res = minimize(fun, dofs, method='BFGS', options={'maxiter': MAXITER}, tol=1e-9)
    elif optimizer == 'dual_annealing':
        initial_temp = 1000
        visit = 2.0
        no_local_search = False
        # bounds = [(np.max([-10*np.abs(dof),-0.21]),np.min([0.21,10*np.abs(dof)])) for dof in dofs]
        bounds = [(-0.25,0.25) for _ in dofs]
        res = dual_annealing(fun, bounds=bounds, maxiter=MAXITER, initial_temp=initial_temp,visit=visit, no_local_search=no_local_search, x0=dofs)
    elif optimizer == 'basinhopping':
        stepsize_minimizer = 0.5
        T_minimizer = 1.0
        res = basinhopping(fun, dofs, niter=MAXITER, stepsize=stepsize_minimizer, T=T_minimizer, disp=True, minimizer_kwargs={"method": "BFGS"})
    elif optimizer =='differential_evolution':
        bounds = [(-0.25,0.25) for _ in dofs]
        # bounds = [(np.max([-10*np.abs(dof),-0.21]),np.min([0.21,10*np.abs(dof)])) for dof in dofs]
        res = differential_evolution(fun, bounds, maxiter=MAXITER, disp=True, x0=dofs)
    elif optimizer == 'least_squares_diff':
        least_squares_mpi_solve(prob, mpi, grad=True, rel_step=diff_rel_step, abs_step=diff_abs_step, max_nfev=MAXITER)
    elif optimizer == 'least_squares':
        least_squares_serial_solve(prob, rel_step=diff_rel_step, abs_step=diff_abs_step, max_nfev=MAXITER)
    if optimizer in ['dual_annealing','minimize','basinhopping','differential_evolution']:
        pprint(f"global minimum: x = {res.x}, f(x) = {res.fun}")
        vmec.x = res.x
    ######################################
    if MPI.COMM_WORLD.rank == 0:
        try: 
            pprint("Final aspect ratio:", vmec.aspect())
            pprint("Final mean iota:", vmec.mean_iota())
            pprint("Final magnetic well:", vmec.vacuum_well())
            pprint("Final mirror ratio:", MirrorRatioPen(v=vmec, output_mirror=True))
            pprint("Final max elongation:", MaxElongationPen(vmec=vmec, return_elongation=True))
            pprint("Quasisymmetry objective after optimization:", qs.total())
            B_scale = 5.7/vmec.wout.b0/redux_B  # Scale the magnetic field by a factor
            Aminor_scale = 1.7/vmec.wout.Aminor_p/redux_Aminor  # Scale the machine size by a factor
            g_field = Simple(wout_filename=vmec.output_file, B_scale=B_scale, Aminor_scale=Aminor_scale,multharm=multharm,ns_s=ns_s,ns_tp=ns_tp)
            g_orbits = ParticleEnsembleOrbit_Simple(g_particle,g_field,tfinal=tfinal,nparticles=nparticles,notrace_passing=notrace_passing,nper=nper,npoiper=npoiper,npoiper2=npoiper2)
            pprint("Final loss fraction:", g_orbits.total_particles_lost)
            pprint("Total objective after optimization:", prob.objective())
        except Exception as e: pprint(e)
    ######################################
if MPI.COMM_WORLD.rank == 0:
    try:
        for objective_file in glob.glob("objective_*"):
            os.remove(objective_file)
        for residuals_file in glob.glob("residuals_*"):
            os.remove(residuals_file)
        for jac_file in glob.glob("jac_log_*"):
            os.remove(jac_file)
        for threed_file in glob.glob("threed1.*"):
            os.remove(threed_file)
        for threed_file in glob.glob("wout_*"):
            os.remove(threed_file)
        for threed_file in glob.glob("input.*"):
            os.remove(threed_file)
        os.remove('fort.6601')
    except Exception as e:
        pprint(e)
    ##################################################
    vmec.write_input(os.path.join(OUT_DIR, f'input.final'))
######################################
if plot_result and MPI.COMM_WORLD.rank==0:
    vmec_final = Vmec(os.path.join(OUT_DIR, f'input.final'), mpi=mpi)
    vmec_final.indata.ns_array[:3]    = [  16,    51,    101]#,   151,   201]
    vmec_final.indata.niter_array[:3] = [ 4000, 10000,  4000]#,  5000, 10000]
    vmec_final.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-14]#, 1e-15, 1e-15]
    vmec_final.run()
    shutil.move(os.path.join(OUT_DIR, f"wout_final_000_000000.nc"), os.path.join(OUT_DIR, f"wout_final.nc"))
    os.remove(os.path.join(OUT_DIR, f'input.final_000_000000'))
    try: vmecPlot2.main(file=os.path.join(OUT_DIR, f"wout_final.nc"), name='EP_opt', figures_folder=OUT_DIR)
    except Exception as e: print(e)
    pprint('Creating Boozer class for vmec_final')
    b1 = Boozer(vmec_final, mpol=64, ntor=64)
    boozxform_nsurfaces=10
    pprint('Defining surfaces where to compute Boozer coordinates')
    booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
    pprint(f' booz_surfaces={booz_surfaces}')
    b1.register(booz_surfaces)
    pprint('Running BOOZ_XFORM')
    b1.run()
    b1.bx.write_boozmn(os.path.join(OUT_DIR,"boozmn_single_stage.nc"))
    pprint("Plot BOOZ_XFORM")
    fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_1_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_2_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_3_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.symplot(b1.bx, helical_detail = True, sqrts=True)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_symplot_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_modeplot_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
############################################################################
############################################################################