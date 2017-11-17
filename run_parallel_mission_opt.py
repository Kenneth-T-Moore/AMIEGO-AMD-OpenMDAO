from mpi4py import MPI
import numpy as np
import pickle
import os

from amd_om.allocation.airline_networks.prob_128_4 import allocation_data
from amd_om.allocation.airline_networks.general_allocation_data import general_allocation_data

from amd_om.mission_analysis.components.aerodynamics.rans_3d_data import get_aero_smt_model, get_rans_crm_wing
from amd_om.mission_analysis.components.propulsion.b777_engine_data import get_prop_smt_model
from amd_om.mission_analysis.utils.plot_utils import plot_single_mission_altitude, plot_single_mission_data
from amd_om.mission_analysis.mission_group import perform_mission_opt

from amd_om.utils.aircraft_data.CRM_full_scale import get_aircraft_data


this_dir = os.path.split(__file__)[0]
output_dir = this_dir + '_mission_outputs/'

training_data_filename = '_design_outputs/training_data.dat'

training_data = np.loadtxt(os.path.join(this_dir, training_data_filename))
CLt = training_data[:, 1]
CDt = training_data[:, 0]

aircraft_data = get_aircraft_data()

ref_area_m2 = aircraft_data['areaRef_m2']
Wac_1e6_N = aircraft_data['Wac_1e6_N']
Wpax_N = general_allocation_data['weight_pax_N']
Mach_mode = 'TAS'

propulsion_model = get_prop_smt_model()
aerodynamics_model = get_aero_smt_model()

xt, yt, xlimits = get_rans_crm_wing()
aerodynamics_model.xt = xt

num_routes = allocation_data['num']
num_existing_aircraft = allocation_data['num_existing']

comm = MPI.COMM_WORLD
num_procs = comm.size
iproc = comm.rank

assert num_procs == num_routes, 'must run with num_procs = num_routes'
split_comm = comm.Split(color=iproc)

# A hack to make parallel optimizations work
MPI.COMM_WORLD = split_comm

mission_index = iproc

num_control_points = int(allocation_data['num_cp'][mission_index])
num_points = int(allocation_data['num_pt'][mission_index])
range_1e3_km = allocation_data['range_km'][mission_index] / 1.e3
output_data_filename = 'optimum_msn_{:03}.pkl'.format(mission_index)
perform_mission_opt(CLt, CDt, output_dir, output_data_filename, comm=split_comm,
    num_control_points=num_control_points, num_points=num_points, range_1e3_km=range_1e3_km,
    ref_area_m2=ref_area_m2, Wac_1e6_N=Wac_1e6_N, Wpax_N=Wpax_N, Mach_mode=Mach_mode,
    mission_index=mission_index, num_existing_aircraft=num_existing_aircraft,
    propulsion_model=propulsion_model, aerodynamics_model=aerodynamics_model,
)
