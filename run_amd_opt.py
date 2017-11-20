import numpy as np
import pickle
import os

from prob_11_2_updated import allocation_data
from amd_om.allocation.airline_networks.general_allocation_data import general_allocation_data
from amd_om.allocation_mission import AllocationMissionGroup

from amd_om.design.utils.flight_conditions import get_flight_conditions

from amd_om.mission_analysis.components.aerodynamics.rans_3d_data import get_aero_smt_model, get_rans_crm_wing
from amd_om.mission_analysis.components.propulsion.b777_engine_data import get_prop_smt_model
from amd_om.allocation_mission_design import AllocationMissionDesignGroup
from amd_om.mission_analysis.utils.plot_utils import plot_single_mission_altitude, plot_single_mission_data

from amd_om.utils.aircraft_data.CRM_full_scale import get_aircraft_data
from amd_om.utils.pre_setup import aeroOptions, meshOptions


def perform_allocation_mission_design_opt(initial_dvs, output_dir, record, **kwargs):
    from six import iteritems

    from openmdao.api import Problem
    from openmdao.parallel_api import PETScVector

    from amd_om.utils.pyoptsparse_setup import get_pyoptsparse_driver
    from amd_om.utils.recorder_setup import get_recorder

    list_of_kwargs = [
        'flight_conditions', 'aircraft_data', 'aeroOptions', 'meshOptions', 'design_variables',
        'general_allocation_data', 'allocation_data', 'ref_area_m2', 'Wac_1e6_N', 'Mach_mode',
        'propulsion_model', 'aerodynamics_model'
    ]
    for key in list_of_kwargs:
        if key not in kwargs:
            raise Exception('missing key:', key)

    prob = Problem(model=AllocationMissionDesignGroup(**kwargs))

    snopt_file_name = 'SNOPT_print_amd.out'
    recorder_file_name = 'recorder_amd.db'

    prob.driver = get_pyoptsparse_driver()
    prob.driver.opt_settings['Print file'] = os.path.join(output_dir, snopt_file_name)

    system_includes = []
    system_includes.append('design_group.concatenating_comp.CLt')
    system_includes.append('design_group.concatenating_comp.CDt')
    for ind in range(128):
        msn_name = 'allocation_mission_group.multi_mission_group.mission_{}'.format(ind)
        system_includes.append(msn_name + '.functionals.fuelburn_comp.fuelburn_1e6_N')
        system_includes.append(msn_name + '.functionals.blocktime_comp.blocktime_hr')
        system_includes.append(msn_name + '.bsplines.comp_x.x_1e3_km')
        system_includes.append(msn_name + '.bsplines.comp_h.h_km')
        system_includes.append(msn_name + '.atmos.mach_number_comp.M')
        system_includes.append(msn_name + '.sys_coupled_analysis.vertical_eom_comp.CL')
        system_includes.append(msn_name + '.sys_coupled_analysis.aero_comp.CD')

    if record:
        recorder = get_recorder(os.path.join(output_dir, recorder_file_name))
        prob.driver.add_recorder(recorder)
        prob.driver.recording_options['system_includes'] = system_includes

    prob.setup(vector_class=PETScVector)

    for key, value in iteritems(initial_dvs):
        prob[key] = value

    prob.run_model()

    prob.run_driver()


this_dir = os.path.split(__file__)[0]
output_dir = this_dir + '_amd_outputs/'

flight_conditions = get_flight_conditions()

aeroOptions = {'gridFile' : '../../../grids/L3_myscaled.cgns',
               'writesurfacesolution' : False,
               'writevolumesolution' : True,
               'writetecplotsurfacesolution' : False,
               'grad_scaler' : 10.,
               }
meshOptions = {'gridFile' : '../../../grids/L3_myscaled.cgns'}

record = True

design_variables = ['shape', 'twist', 'sweep', 'area']

initial_dvs = {}

optimum_design_filename = '_design_outputs/optimum_design.pkl'
optimum_design_data = pickle.load(open(os.path.join(this_dir, optimum_design_filename), 'rb'))
for key in ['shape', 'twist', 'sweep', 'area']:
    initial_dvs[key] = optimum_design_data[key]

optimum_alloc_filename = '_allocation_outputs/optimum_alloc.pkl'
optimum_alloc_data = pickle.load(open(os.path.join(this_dir, optimum_alloc_filename), 'rb'))
for key in ['pax_flt', 'flt_day']:
    initial_dvs[key] = optimum_alloc_data[key]

initial_mission_vars = {}

num_routes = allocation_data['num']
for ind in range(num_routes):
    optimum_mission_filename = '_mission_outputs/optimum_msn_{:03}.pkl'.format(ind)
    optimum_mission_data = pickle.load(open(os.path.join(this_dir, optimum_mission_filename), 'rb'))
    for key in ['h_km_cp', 'M0']:
        initial_mission_vars[ind, key] = optimum_mission_data[key]

aircraft_data = get_aircraft_data()

ref_area_m2 = aircraft_data['areaRef_m2']
Wac_1e6_N = aircraft_data['Wac_1e6_N']
Mach_mode = 'TAS'

propulsion_model = get_prop_smt_model()
aerodynamics_model = get_aero_smt_model()

xt, yt, xlimits = get_rans_crm_wing()
aerodynamics_model.xt = xt

perform_allocation_mission_design_opt(initial_dvs, output_dir, record,
    flight_conditions=flight_conditions, aircraft_data=aircraft_data,
    aeroOptions=aeroOptions, meshOptions=meshOptions, design_variables=design_variables,
    general_allocation_data=general_allocation_data, allocation_data=allocation_data,
    ref_area_m2=ref_area_m2, Wac_1e6_N=Wac_1e6_N, Mach_mode=Mach_mode,
    propulsion_model=propulsion_model, aerodynamics_model=aerodynamics_model,
    initial_mission_vars=initial_mission_vars,
)

