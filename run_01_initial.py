"""
Run Amiego plus AMD
"""
import pickle
import os

import numpy as np

from six import iteritems

from openmdao.api import Problem, Group, IndepVarComp, NonlinearBlockGS, LinearBlockGS, ExecComp
from openmdao.parallel_api import PETScVector
from openmdao.utils.mpi import MPI

from amd_om.design.design_group import DesignGroup
from amd_om.design.utils.flight_conditions import get_flight_conditions
from amd_om.mission_analysis.components.aerodynamics.rans_3d_data import get_aero_smt_model, get_rans_crm_wing
from amd_om.mission_analysis.components.propulsion.b777_engine_data import get_prop_smt_model
from amd_om.mission_analysis.multi_mission_group import MultiMissionGroup
from amd_om.mission_analysis.utils.plot_utils import plot_single_mission_altitude, plot_single_mission_data
from amd_om.utils.aircraft_data.CRM_full_scale import get_aircraft_data
from amd_om.utils.recorder_setup import get_recorder

from amd_om.utils.pre_setup import aeroOptions, meshOptions

from amiego_pre_opt import AMIEGO_With_Pre
from economy import Profit, RevenueManager
from prob_11_2_updated import allocation_data
from prob_11_2_general_allocation import general_allocation_data
from preopt.load_preopt import load_all_preopts
from preopt_screen import pyOptSparseWithScreening


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
            aircraft_name =  allocation_data['names'][ind_ac]
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

        self.add_design_var('flt_day', lower=flt_day_lower, upper=flt_day_upper)
        self.add_design_var('revenue:x1', lower=0.0, ref=1.0e3)
        self.add_design_var('revenue:y1', lower=0.0, ref=1.0e3)
        self.add_design_var('revenue:x2', lower=0.0, ref=1.0e3)
        self.add_design_var('revenue:y2', lower=0.0, ref=1.0e3)
        self.add_design_var('revenue:z1', lower=0.0, ref=1.0e3)
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

        design_group = DesignGroup(
            flight_conditions=flight_conditions, aircraft_data=aircraft_data,
            aeroOptions=aeroOptions, meshOptions=meshOptions, design_variables=design_variables,
        )
        self.add_subsystem('design_group', design_group, promotes=['*'])

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

new_aeroOptions = {'gridFile' : '../Plugins/amd_om/grids/L3_myscaled.cgns',
                   'writesurfacesolution' : False,
                   'writevolumesolution' : True,
                   'writetecplotsurfacesolution' : False,
                   'grad_scaler' : 10.,
                   'outputDirectory' : grid_dir,
                   }

new_meshOptions = {'gridFile' : '../Plugins/amd_om/grids/L3_myscaled.cgns'}

record = True

design_variables = ['shape', 'twist', 'sweep', 'area']

initial_dvs = {}

# Loads in initial design variables for aero.
optimum_design_filename = './_design_outputs/optimum_design.pkl'
optimum_design_data = pickle.load(open(optimum_design_filename, 'rb'))
#optimum_design_data = pickle.load(open(os.path.join(this_dir, optimum_design_filename), 'rb'))
for key in ['shape', 'twist', 'sweep', 'area']:
    initial_dvs[key] = optimum_design_data[key]

initial_mission_vars = {}

num_routes = allocation_data['num']

initial_mission_vars[0, 'M0'] =  np.array([0.864429])
initial_mission_vars[0, 'h_km_cp'] =  np.array([ 0.      , 10.35209 ,  9.637666,  9.801114,  9.731797,  9.740211,
        9.772475,  9.886949, 10.056549, 10.374795, 10.694507, 11.107989,
       11.56719 , 11.743023, 12.014062, 12.059805, 12.300495, 12.973339,
        2.77534 ,  0.      ])
initial_mission_vars[1, 'M0'] =  np.array([0.865])
initial_mission_vars[1, 'h_km_cp'] =  np.array([ 0.      ,  2.283885,  9.517857, 11.718341, 11.471231, 11.710785,
       11.696944, 11.810669, 11.862983, 11.954126, 12.010739, 12.107551,
       12.123627, 12.282498, 12.119328, 12.894419,  7.384425,  3.297598,
        0.801587,  0.      ])
initial_mission_vars[10, 'M0'] =  np.array([0.865])
initial_mission_vars[10, 'h_km_cp'] =  np.array([ 0.      ,  4.170787, 12.040081, 10.151405, 11.027536, 10.816799,
       11.136823, 11.292572, 11.504891, 11.638092, 11.770833, 11.906133,
       11.990681, 12.146116, 12.121515, 12.388223, 12.415268,  5.318532,
        1.356051,  0.      ])
initial_mission_vars[2, 'M0'] =  np.array([0.863749])
initial_mission_vars[2, 'h_km_cp'] =  np.array([ 0.      ,  3.353335, 11.790452, 10.563563, 11.212514, 11.124893,
       11.396046, 11.493976, 11.629653, 11.752384, 11.837858, 12.002475,
       11.976665, 12.304207, 11.890392, 12.959757, 10.930736,  4.593093,
        1.188606,  0.      ])
initial_mission_vars[3, 'M0'] =  np.array([0.865])
initial_mission_vars[3, 'h_km_cp'] =  np.array([ 0.      ,  0.787903,  3.142244,  6.627033, 10.369104, 12.33404 ,
       12.274079, 12.199388, 12.202051, 12.217892, 12.278023, 12.275476,
       12.530899, 10.15307 ,  7.172636,  4.637384,  2.579766,  1.165995,
        0.28254 ,  0.      ])
initial_mission_vars[4, 'M0'] =  np.array([0.813061])
initial_mission_vars[4, 'h_km_cp'] =  np.array([ 0.      , 10.226025, 10.129094,  9.866376, 10.009642, 10.11874 ,
       10.341841, 10.502426, 10.753355, 11.008159, 11.285974, 11.495762,
       11.733091, 11.784829, 12.156365, 11.840942, 12.801616, 11.372046,
        2.050718,  0.      ])
initial_mission_vars[5, 'M0'] =  np.array([0.795156])
initial_mission_vars[5, 'h_km_cp'] =  np.array([ 0.      ,  0.611339,  2.436113,  5.258926,  8.651994, 11.545463,
       13.      , 13.      , 13.      , 12.522025, 12.423772, 12.184545,
       10.441223,  7.778774,  5.517264,  3.505857,  1.979039,  0.883492,
        0.217632,  0.      ])
initial_mission_vars[6, 'M0'] =  np.array([0.858225])
initial_mission_vars[6, 'h_km_cp'] =  np.array([ 0.      ,  2.214406,  9.219697, 11.796525, 11.478195, 11.736852,
       11.711025, 11.826356, 11.876557, 11.961469, 12.026964, 12.099238,
       12.159559, 12.225116, 12.248215, 12.643896,  7.20061 ,  3.22206 ,
        0.781395,  0.      ])
initial_mission_vars[7, 'M0'] =  np.array([0.865])
initial_mission_vars[7, 'h_km_cp'] =  np.array([ 0.      ,  2.414538, 10.050323, 11.546091, 11.465041, 11.652454,
       11.670448, 11.776194, 11.837535, 11.933405, 11.988602, 12.102384,
       12.093759, 12.312198, 12.036121, 13.      ,  7.88779 ,  3.409916,
        0.860561,  0.      ])
initial_mission_vars[8, 'M0'] =  np.array([0.863127])
initial_mission_vars[8, 'h_km_cp'] =  np.array([ 0.      ,  1.408406,  5.629408, 10.625147, 12.187488, 11.842597,
       12.030189, 11.99636 , 12.076522, 12.104191, 12.156514, 12.200775,
       12.233027, 12.303459, 12.487821,  8.175946,  4.632962,  2.043096,
        0.503667,  0.      ])
initial_mission_vars[9, 'M0'] =  np.array([0.864998])
initial_mission_vars[9, 'h_km_cp'] =  np.array([ 0.      ,  5.336129, 11.675169,  9.792977, 10.695585, 10.43858 ,
       10.818874, 10.901298, 11.211396, 11.45758 , 11.641887, 11.780121,
       11.951703, 12.010029, 12.239684, 12.042562, 12.999989,  7.039606,
        1.486818,  0.      ])

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
recorder_file_name = 'recorder_amd.db'


print("Running Setup")
prob.setup(vector_class=PETScVector)
print("Setup Complete")

# Insert Optimal Design here

for key, value in iteritems(initial_dvs):
    prob[key] = value

prob['flt_day'] = np.array([[  0.,   0.,   0.,   0.,  12.,   3.,   4.,   0.,   6.,   0.,   0.,   5.,   0.,   0.,   2.,
                               1.,   0.,   1.,   1.,   1.,   1.,   0.,   0.,   4.,   4.,   4.,   0.,   0.,   0.,   1.,
                               0.,   0.,   1.]]).reshape(11, 3)


prob['revenue:x1'] = np.array(([4.346   , 1.343   , 1.756   , 0.706166, 3.657   , 0.618897,
       1.32    , 1.389   , 0.980999, 2.425   , 1.965   ]),) * 1.0e3
prob['revenue:x2'] =  np.array([6.021588, 1.16837 , 1.553047, 0.630359, 4.841167, 0.678703,
       1.140274, 1.201012, 0.85849 , 2.107893, 1.676628]) * 1.0e3
prob['revenue:y1'] =  np.array([2.569894, 0.450041, 0.588163, 0.235874, 1.224314, 0.203671,
       0.441217, 0.463736, 0.332634, 0.814729, 0.664564]) * 1.0e3
prob['revenue:y2'] =  np.array([3.651259, 0.652964, 0.867739, 0.332958, 2.583519, 0.322815,
       0.636901, 0.671654, 0.476217, 1.169473, 0.937639]) * 1.0e3
prob['revenue:z1'] =  np.array([0.3  , 3.192, 4.112, 2.242, 1.248, 0.3  , 0.716, 1.896, 2.264,
       0.416, 0.416]) * 1.0e3

prob.run_model()

prob.model.allocation_mission_group.allocation_group.profit_comp.list_outputs()
prob.model.allocation_mission_group.revenue_comp.list_outputs()

for name in ['flt_day', 'revenue:x1', 'revenue:x2', 'revenue:y1', 'revenue:y2', 'revenue:z1', 'costs', 'revenue', 'tot_pax', 'pax_flt', 'nacc', 'p', 'profit']:
    print(name, prob[name])


print(prob.driver.get_design_var_values())