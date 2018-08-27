"""
Run Amiego plus AMD
"""
import pickle
import os

import numpy as np

from six import iteritems

from openmdao.api import Problem, Group, IndepVarComp, NonlinearBlockGS, LinearBlockGS, ExecComp
from openmdao.api import ParallelGroup, NonlinearBlockJac, LinearBlockJac, pyOptSparseDriver
from openmdao.drivers.amiego_driver import AMIEGO_driver
from openmdao.parallel_api import PETScVector
from openmdao.utils.mpi import MPI

from amd_om.design.utils.flight_conditions import get_flight_conditions
from amd_om.mission_analysis.components.aerodynamics.rans_3d_data import get_aero_smt_model, get_rans_crm_wing
from amd_om.mission_analysis.components.propulsion.b777_engine_data import get_prop_smt_model
from amd_om.mission_analysis.mission_group import MissionGroup
from amd_om.mission_analysis.utils.plot_utils import plot_single_mission_altitude, plot_single_mission_data
from amd_om.utils.aircraft_data.CRM_full_scale import get_aircraft_data
from amd_om.utils.recorder_setup import get_recorder

from amiego_pre_opt import AMIEGO_With_Pre
from economy import Profit, RevenueManager
from prob_11_2_updated import allocation_data
from prob_11_2_general_allocation import general_allocation_data
from preopt.load_preopt import load_all_preopts
from preopt_screen import pyOptSparseWithScreening


class MultiMissionGroup(ParallelGroup):

    def initialize(self):
        self.metadata.declare('general_allocation_data', types=dict)
        self.metadata.declare('allocation_data', types=dict)

        self.metadata.declare('ref_area_m2', default=10., types=np.ScalarType)
        self.metadata.declare('Wac_1e6_N', default=1., types=np.ScalarType)
        self.metadata.declare('Mach_mode', 'TAS', values=['TAS', 'EAS', 'IAS', 'constant'])

        self.metadata.declare('propulsion_model')
        self.metadata.declare('aerodynamics_model')

        self.metadata.declare('initial_mission_vars', types=dict, allow_none=True)

    def setup(self):
        meta = self.metadata

        general_allocation_data = meta['general_allocation_data']
        allocation_data = meta['allocation_data']

        ref_area_m2 = meta['ref_area_m2']
        Wac_1e6_N = meta['Wac_1e6_N']
        Wpax_N = general_allocation_data['weight_pax_N']
        Mach_mode = meta['Mach_mode']

        propulsion_model = meta['propulsion_model']
        aerodynamics_model = meta['aerodynamics_model']

        initial_mission_vars = meta['initial_mission_vars']

        num_routes = allocation_data['num']

        for ind_r in range(num_routes):
            num_points = int(allocation_data['num_pt'][ind_r])
            num_control_points = int(allocation_data['num_cp'][ind_r])
            range_1e3_km = allocation_data['range_km'][ind_r] / 1e3

            self.add_subsystem('mission_{}'.format(ind_r),
                MissionGroup(
                    num_control_points=num_control_points, num_points=num_points,
                    range_1e3_km=range_1e3_km, ref_area_m2=ref_area_m2,
                    Wac_1e6_N=Wac_1e6_N, Wpax_N=Wpax_N, Mach_mode=Mach_mode,
                    mission_index=ind_r, propulsion_model=propulsion_model,
                    aerodynamics_model=aerodynamics_model,
                    initial_mission_vars=initial_mission_vars,
                ),
                promotes=['pax_flt', 'CLt', 'CDt'],
            )

        self.nonlinear_solver = NonlinearBlockJac()
        self.nonlinear_solver.options['maxiter'] = 1
        self.linear_solver = LinearBlockJac()
        self.linear_solver.options['maxiter'] = 1


class AllocationGroup(Group):
    """
    Modified version of AllocationGroup that includes part of Revenuue Management.
    """
    def initialize(self):
        self.metadata.declare('general_allocation_data', types=dict)
        self.metadata.declare('allocation_data', types=dict)

    def setup(self):
        general_allocation_data = self.metadata['general_allocation_data']
        allocation_data = self.metadata['allocation_data']

        num_routes = allocation_data['num']
        num_existing_aircraft = allocation_data['num_existing']
        num_new_aircraft = allocation_data['num_new']
        num_aircraft = num_existing_aircraft + num_new_aircraft

        self.add_subsystem('profit_comp', Profit(general_allocation_data=general_allocation_data,
                                                 allocation_data=allocation_data
                                                 ), promotes=['*'])

        # Add Constraints
        self.add_constraint('g_aircraft_new',
                            upper=1.0,
                            vectorize_derivs=True,
                            parallel_deriv_color='pdc:demand_constraint',
        )

        self.add_constraint('g_aircraft_exist',
                            upper=1.0,
                            vectorize_derivs=True,
                            parallel_deriv_color='pdc:demand_constraint',
        )

        self.add_constraint('g_demand',
                            upper=1.0,
                            vectorize_derivs=True,
                            parallel_deriv_color='pdc:demand_constraint',
        )


class AllocationMissionGroup(Group):

    def initialize(self):
        self.metadata.declare('general_allocation_data', types=dict)
        self.metadata.declare('allocation_data', types=dict)

        self.metadata.declare('ref_area_m2', default=10., types=np.ScalarType)
        self.metadata.declare('Wac_1e6_N', default=1., types=np.ScalarType)
        self.metadata.declare('Mach_mode', 'TAS', values=['TAS', 'EAS', 'IAS', 'constant'])

        self.metadata.declare('propulsion_model')
        self.metadata.declare('aerodynamics_model')

        self.metadata.declare('initial_mission_vars', default=None, types=dict, allow_none=True)

    def setup(self):
        meta = self.metadata

        general_allocation_data = meta['general_allocation_data']
        allocation_data = meta['allocation_data']

        ref_area_m2 = meta['ref_area_m2']
        Wac_1e6_N = meta['Wac_1e6_N']
        Mach_mode = meta['Mach_mode']

        propulsion_model = meta['propulsion_model']
        aerodynamics_model = meta['aerodynamics_model']

        initial_mission_vars = meta['initial_mission_vars']

        allocation_data = meta['allocation_data']

        num_routes = allocation_data['num']
        num_existing_aircraft = allocation_data['num_existing']
        num_new_aircraft = allocation_data['num_new']
        num_aircraft = num_existing_aircraft + num_new_aircraft

        def get_ones_array(val):
            return val * np.ones((num_routes, num_aircraft))

        flt_day = get_ones_array(1e-2)

        flt_day_lower = get_ones_array( 0.)
        flt_day_upper = get_ones_array(0.)

        seats = []
        for key in allocation_data['names']:
            seats.append(allocation_data['capacity', key])
        seats = np.array(seats)

        for ind_ac in range(num_aircraft):
            aircraft_name = allocation_data['names'][ind_ac]
            for ind_rt in range(num_routes):

                # Zeroing out the planes that can't fly that far.
                key = ('fuel_N', aircraft_name)
                if key in allocation_data and allocation_data[key][ind_rt] > 1e12:
                    flt_day[ind_rt, ind_ac] = 0.
                    flt_day_lower[ind_rt, ind_ac] = 0.
                    flt_day_upper[ind_rt, ind_ac] = 0.
                else:
                    # Setting the upper bound
                    flt_day_upper[ind_rt, ind_ac] = np.ceil(allocation_data['demand'][ind_rt]/(0.8*seats[ind_ac]))

        # Revenue Initial Conditions:
        xC0_rev = 1.0e3*np.array([[ 4.3460,    1.3430,    1.7560,    0.7062,    3.6570,    0.6189,    1.3200,    1.3890,    0.9810,    2.4250,    1.9650],
                                  [ 2.6318,    0.4475,    0.5851,    0.2357,    1.2197,    0.2159,    0.4400,    0.4629,    0.3269,    0.7980,    0.6416],
                                  [ 6.0180,    1.1577,    1.5140,    0.6311,    3.1820,    0.7101,    1.1561,    1.2043,    0.8479,    2.1036,    1.6272],
                                  [ 3.7277,    0.6471,    0.8467,    0.3331,    1.7368,    0.3450,    0.6442,    0.6730,    0.4673,    1.1667,    0.9138],
                                  [ 0.3000,    3.1920,    4.1120,    2.2420,    1.2480,    0.3000,    0.7160,    1.8960,    2.2640,    0.4160,    0.4160]])

        inputs_comp = IndepVarComp()
        inputs_comp.add_output('flt_day', val=flt_day, shape=(num_routes, num_aircraft))
        inputs_comp.add_output('revenue:x1', val=xC0_rev[0, :], shape=(num_routes, ))
        inputs_comp.add_output('revenue:y1', val=xC0_rev[1, :], shape=(num_routes, ))
        inputs_comp.add_output('revenue:x2', val=xC0_rev[2, :], shape=(num_routes, ))
        inputs_comp.add_output('revenue:y2', val=xC0_rev[3, :], shape=(num_routes, ))
        inputs_comp.add_output('revenue:z1', val=xC0_rev[4, :], shape=(num_routes, ))

        revenue_comp = RevenueManager(general_allocation_data=general_allocation_data,
                                      allocation_data=allocation_data)

        multi_mission_group = MultiMissionGroup(
            general_allocation_data=general_allocation_data, allocation_data=allocation_data,
            ref_area_m2=ref_area_m2, Wac_1e6_N=Wac_1e6_N, Mach_mode=Mach_mode,
            propulsion_model=propulsion_model, aerodynamics_model=aerodynamics_model,
            initial_mission_vars=initial_mission_vars,
        )

        allocation_group = AllocationGroup(
            general_allocation_data=general_allocation_data, allocation_data=allocation_data,
        )

        self.add_subsystem('inputs_comp', inputs_comp, promotes=['*'])
        self.add_subsystem('revenue_comp', revenue_comp, promotes=['*'])
        self.add_subsystem('multi_mission_group', multi_mission_group, promotes=['*'])
        self.add_subsystem('allocation_group', allocation_group, promotes=['*'])

        for ind in range(allocation_data['num']):
            self.connect(
                'mission_{}.fuelburn_1e6_N'.format(ind),
                '{}_fuelburn_1e6_N'.format(ind),
            )
            self.connect(
                'mission_{}.blocktime_hr'.format(ind),
                '{}_blocktime_hr'.format(ind),
            )

        demand_constraint = np.zeros((num_routes, num_aircraft))
        for ind_ac in range(num_aircraft):
            aircraft_name = allocation_data['names'][ind_ac]
            for ind_rt in range(num_routes):
                demand_constraint[ind_rt, ind_ac] = allocation_data['capacity', aircraft_name]

        #self.add_design_var('flt_day', lower=flt_day_lower, upper=flt_day_upper)
        self.add_objective('profit')


class AllocationMissionDesignGroup(Group):

    def initialize(self):
        self.metadata.declare('flight_conditions', types=dict)
        self.metadata.declare('aircraft_data', types=dict)
        self.metadata.declare('aeroOptions', default=None, types=dict, allow_none=True)
        self.metadata.declare('meshOptions', default=None, types=dict, allow_none=True)
        self.metadata.declare('design_variables', types=list,
            default=['shape', 'twist', 'sweep', 'area'])

        self.metadata.declare('general_allocation_data', types=dict)
        self.metadata.declare('allocation_data', types=dict)

        self.metadata.declare('ref_area_m2', default=10., types=np.ScalarType)
        self.metadata.declare('Wac_1e6_N', default=1., types=np.ScalarType)
        self.metadata.declare('Mach_mode', 'TAS', values=['TAS', 'EAS', 'IAS', 'constant'])

        self.metadata.declare('propulsion_model')
        self.metadata.declare('aerodynamics_model')

        self.metadata.declare('initial_mission_vars', types=dict, allow_none=True)

    def setup(self):
        meta = self.metadata

        flight_conditions = meta['flight_conditions']
        aircraft_data = meta['aircraft_data']
        design_variables = meta['design_variables']

        if meta['aeroOptions']:
            aeroOptions.update(meta['aeroOptions'])

        if meta['meshOptions']:
            meshOptions.update(meta['meshOptions'])

        general_allocation_data = meta['general_allocation_data']
        allocation_data = meta['allocation_data']

        ref_area_m2 = meta['ref_area_m2']
        Wac_1e6_N = meta['Wac_1e6_N']
        Mach_mode = meta['Mach_mode']

        propulsion_model = meta['propulsion_model']
        aerodynamics_model = meta['aerodynamics_model']

        initial_mission_vars = meta['initial_mission_vars']

        allocation_mission_group = AllocationMissionGroup(
            general_allocation_data=general_allocation_data, allocation_data=allocation_data,
            ref_area_m2=ref_area_m2, Wac_1e6_N=Wac_1e6_N, Mach_mode=Mach_mode,
            propulsion_model=propulsion_model, aerodynamics_model=aerodynamics_model,
            initial_mission_vars=initial_mission_vars,
        )
        self.add_subsystem('allocation_mission_group', allocation_mission_group, promotes=['*'])


#-------------------------------------------------------------------------

# Give meshes their own directory based on proc rank.
grid_dir = '/nobackupp2/ktmoore1/run1'
if MPI:
    rank = MPI.COMM_WORLD.rank
    grid_dir += '/' + str(rank)

    # Make sure path exists:
    if not os.path.isdir(grid_dir):
        os.makedirs(grid_dir)


this_dir = os.path.split(__file__)[0]
if not this_dir.endswith('/'):
    this_dir += '/'
output_dir = './_amd_outputs/'

flight_conditions = get_flight_conditions()

new_aeroOptions = {}

new_meshOptions = {}

record = True

design_variables = ['shape', 'twist', 'sweep', 'area']

initial_dvs = {}
initial_mission_vars = {}

initial_mission_vars[0, 'M0'] =  np.array([0.864767])
initial_mission_vars[0, 'h_km_cp'] =  np.array([ 0.      , 10.35209 ,  9.637666,  9.801114,  9.731797,  9.740211,
                                                 9.772475,  9.886949, 10.056549, 10.374795, 10.694507, 11.107989,
                                                 11.56719 , 11.743023, 12.014062, 12.059805, 12.300756, 12.993253,
                                                 2.845975,  0.      ])
initial_mission_vars[1, 'M0'] =  np.array([0.865])
initial_mission_vars[1, 'h_km_cp'] =  np.array([ 0.      ,  2.283885,  9.517857, 11.718341, 11.471231, 11.710785,
                                                 11.696944, 11.810669, 11.862983, 11.954126, 12.010739, 12.107551,
                                                 12.123627, 12.282498, 12.119328, 12.894419,  7.384425,  3.297598,
                                                 0.801587,  0.      ])
initial_mission_vars[10, 'M0'] =  np.array([0.865])
initial_mission_vars[10, 'h_km_cp'] =  np.array([ 0.      ,  4.170825, 12.040107, 10.151466, 11.027615, 10.816905,
                                                  11.136938, 11.292667, 11.50498 , 11.638173, 11.770906, 11.906197,
                                                  11.990738, 12.14617 , 12.12156 , 12.388267, 12.415673,  5.319076,
                                                  1.356266,  0.      ])
initial_mission_vars[2, 'M0'] =  np.array([0.865])
initial_mission_vars[2, 'h_km_cp'] =  np.array([ 0.      ,  3.353327, 11.790421, 10.563483, 11.212765, 11.125226,
                                                 11.396441, 11.494415, 11.630118, 11.752858, 11.838323, 12.002917,
                                                 11.977072, 12.304558, 11.890665, 12.959723, 10.926817,  4.589707,
                                                 1.189852,  0.      ])
initial_mission_vars[3, 'M0'] =  np.array([0.865])
initial_mission_vars[3, 'h_km_cp'] =  np.array([ 0.      ,  0.787903,  3.142244,  6.627033, 10.369104, 12.33404 ,
                                                 12.274079, 12.199388, 12.202051, 12.217892, 12.278023, 12.275476,
                                                 12.530899, 10.15307 ,  7.172636,  4.637384,  2.579766,  1.165995,
                                                 0.28254 ,  0.      ])
initial_mission_vars[4, 'M0'] =  np.array([0.812785])
initial_mission_vars[4, 'h_km_cp'] =  np.array([ 0.      , 10.225185, 10.128901,  9.866376, 10.009642, 10.11874 ,
                                                 10.341841, 10.502426, 10.753355, 11.008159, 11.285974, 11.495762,
                                                 11.733091, 11.784829, 12.156365, 11.840942, 12.801616, 11.372046,
                                                 2.050718,  0.      ])
initial_mission_vars[5, 'M0'] =  np.array([0.794354])
initial_mission_vars[5, 'h_km_cp'] =  np.array([ 0.      ,  0.611311,  2.436084,  5.25891 ,  8.651973, 11.545441,
                                                 13.      , 13.      , 13.      , 12.522009, 12.423754, 12.18452 ,
                                                 10.441155,  7.778668,  5.517168,  3.505772,  1.978987,  0.883465,
                                                 0.217615,  0.      ])
initial_mission_vars[6, 'M0'] =  np.array([0.865])
initial_mission_vars[6, 'h_km_cp'] =  np.array([ 0.      ,  2.214417,  9.219706, 11.796524, 11.478158, 11.736811,
                                                 11.71098 , 11.826306, 11.876503, 11.961412, 12.026905, 12.099177,
                                                 12.159501, 12.225062, 12.248165, 12.643996,  7.200498,  3.222091,
                                                 0.781408,  0.      ])
initial_mission_vars[7, 'M0'] =  np.array([0.865])
initial_mission_vars[7, 'h_km_cp'] =  np.array([ 0.      ,  2.414538, 10.050323, 11.546091, 11.465041, 11.652454,
                                                 11.670448, 11.776194, 11.837535, 11.933405, 11.988602, 12.102384,
                                                 12.093759, 12.312198, 12.036121, 13.      ,  7.88779 ,  3.409916,
                                                 0.860561,  0.      ])
initial_mission_vars[8, 'M0'] =  np.array([0.865])
initial_mission_vars[8, 'h_km_cp'] =  np.array([ 0.      ,  1.408356,  5.629337, 10.625036, 12.187419, 11.842538,
                                                 12.030119, 11.99628 , 12.076433, 12.104098, 12.15642 , 12.200682,
                                                 12.232939, 12.303391, 12.48715 ,  8.171493,  4.632096,  2.042848,
                                                 0.504083,  0.      ])
initial_mission_vars[9, 'M0'] =  np.array([0.865])
initial_mission_vars[9, 'h_km_cp'] =  np.array([ 0.      ,  5.336129, 11.675169,  9.792977, 10.695585, 10.43858 ,
                                                 10.818874, 10.901298, 11.211396, 11.45758 , 11.641887, 11.780121,
                                                 11.951703, 12.010029, 12.239684, 12.042562, 13.      ,  7.039611,
                                                 1.486823,  0.      ])


aircraft_data = get_aircraft_data()

ref_area_m2 = aircraft_data['areaRef_m2']
Wac_1e6_N = aircraft_data['Wac_1e6_N']
Mach_mode = 'TAS'

propulsion_model = get_prop_smt_model()
aerodynamics_model = get_aero_smt_model()

xt, yt, xlimits = get_rans_crm_wing()
aerodynamics_model.xt = xt


prob = Problem(model=AllocationMissionDesignGroup(flight_conditions=flight_conditions, aircraft_data=aircraft_data,
                                                  aeroOptions=new_aeroOptions, meshOptions=new_meshOptions, design_variables=design_variables,
                                                  general_allocation_data=general_allocation_data, allocation_data=allocation_data,
                                                  ref_area_m2=ref_area_m2, Wac_1e6_N=Wac_1e6_N, Mach_mode=Mach_mode,
                                                  propulsion_model=propulsion_model, aerodynamics_model=aerodynamics_model,
                                                  initial_mission_vars=initial_mission_vars))

snopt_file_name = 'SNOPT_print_amd.out'

#prob.driver = AMIEGO_driver()
#prob.driver.options['disp'] = True
#prob.driver.options['r_penalty'] = 1.0
prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'
prob.driver.opt_settings['Major optimality tolerance'] = 1e3
prob.driver.opt_settings['Major feasibility tolerance'] = 1e3
prob.driver.opt_settings['Print file'] = os.path.join(output_dir, snopt_file_name)
#prob.driver.allocation_data = allocation_data
prob.driver.allocation_data = allocation_data
#prob.driver.minlp.options['trace_iter'] = 5
#prob.driver.minlp.options['maxiter'] = 1

prob.driver.options['debug_print'] = ['desvars']

# Load in initial sample points.
sample_data = np.loadtxt('Initialpoints_AMIEGO_AMD_11rt.dat')
xpose_sample = np.empty(sample_data.shape)
for j in range(sample_data.shape[0]):
    xpose_sample[j, :] = sample_data[j, :].reshape(3, 11).T.flatten()
prob.driver.sampling = {'flt_day' : xpose_sample}


print("Running Setup")
prob.setup(vector_class=PETScVector)

prob.model.allocation_mission_group.multi_mission_group.mission_0.sys_coupled_analysis.nonlinear_solver.options['maxiter'] = 1
prob.model.allocation_mission_group.multi_mission_group.mission_1.sys_coupled_analysis.nonlinear_solver.options['maxiter'] = 1
prob.model.allocation_mission_group.multi_mission_group.mission_2.sys_coupled_analysis.nonlinear_solver.options['maxiter'] = 1
prob.model.allocation_mission_group.multi_mission_group.mission_3.sys_coupled_analysis.nonlinear_solver.options['maxiter'] = 1
prob.model.allocation_mission_group.multi_mission_group.mission_4.sys_coupled_analysis.nonlinear_solver.options['maxiter'] = 1
prob.model.allocation_mission_group.multi_mission_group.mission_5.sys_coupled_analysis.nonlinear_solver.options['maxiter'] = 1
prob.model.allocation_mission_group.multi_mission_group.mission_6.sys_coupled_analysis.nonlinear_solver.options['maxiter'] = 1
prob.model.allocation_mission_group.multi_mission_group.mission_7.sys_coupled_analysis.nonlinear_solver.options['maxiter'] = 1
prob.model.allocation_mission_group.multi_mission_group.mission_8.sys_coupled_analysis.nonlinear_solver.options['maxiter'] = 1
prob.model.allocation_mission_group.multi_mission_group.mission_9.sys_coupled_analysis.nonlinear_solver.options['maxiter'] = 1
prob.model.allocation_mission_group.multi_mission_group.mission_10.sys_coupled_analysis.nonlinear_solver.options['maxiter'] = 1


print("Setup Complete")
for key, value in iteritems(initial_dvs):
    prob[key] = value


prob.run_driver()
prob.run_driver()



