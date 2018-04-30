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
initial_dvs['area'] =  np.array([0.108576]) * 10.0
initial_dvs['shape'] =  np.array([-6.62390e-02,  2.59910e-02,  4.04130e-02,  8.82880e-02,
        4.94810e-02, -6.72620e-02,  1.13366e-01, -1.34294e-01,
       -9.41480e-02, -1.35994e-01, -9.30610e-02,  4.38420e-02,
        1.06671e-01,  6.96070e-02,  2.05470e-02, -3.59390e-02,
       -6.17970e-02, -1.22322e-01,  2.29590e-02,  3.03300e-02,
        4.13780e-02,  6.82000e-02,  6.40240e-02,  3.43320e-02,
        3.06600e-03, -1.39770e-02, -2.84300e-03,  1.97516e-01,
       -6.94600e-03,  2.29930e-02,  3.58930e-02, -2.52400e-03,
       -6.63900e-03, -4.90000e-05,  8.33710e-02,  8.38010e-02,
        1.02892e-01,  1.55625e-01,  1.44184e-01,  1.68600e-01,
        3.32530e-02,  8.95500e-02,  1.15852e-01,  7.64550e-02,
        3.78090e-02,  4.68800e-03, -1.43310e-02,  1.27340e-02,
        1.31620e-02,  5.41730e-02, -3.18010e-02, -1.20490e-02,
        1.21000e-03,  1.63070e-02,  3.33210e-02,  3.31000e-02,
        2.62220e-02, -1.85020e-02, -5.51340e-02, -1.94180e-01,
       -9.71960e-02, -1.47067e-01, -1.04001e-01, -1.06388e-01,
       -1.48863e-01, -1.34030e-01, -1.32589e-01, -3.99040e-02,
       -6.10320e-02, -9.96100e-02, -1.07311e-01, -1.03067e-01,
       -2.76240e-02,  1.14790e-02,  3.16560e-02,  2.56530e-02,
        2.64110e-02,  8.02700e-03, -1.30140e-02,  3.81470e-02,
        5.36200e-02,  4.75790e-02,  4.68260e-02,  3.87900e-02,
        1.32440e-02,  7.59210e-02,  7.75210e-02,  7.09370e-02,
        7.58220e-02,  6.20360e-02,  6.41170e-02,  1.12802e-01,
        1.01167e-01,  8.70670e-02,  8.98730e-02,  7.35630e-02,
        8.71790e-02,  1.38432e-01,  1.11422e-01,  8.83840e-02,
        9.06180e-02,  7.51650e-02,  8.12530e-02,  1.17322e-01,
        9.61680e-02,  7.44860e-02,  7.73760e-02,  5.90350e-02,
        1.05680e-02,  4.30830e-02,  4.32000e-02,  3.81840e-02,
        5.02570e-02,  9.45700e-03, -5.90110e-02, -3.06800e-02,
       -1.86120e-02,  7.35300e-03,  1.46100e-02, -2.56840e-02,
       -9.89740e-02, -9.13530e-02, -5.23400e-02, -2.10810e-02,
       -1.61280e-02, -2.55760e-02, -6.72790e-02,  9.21970e-02,
        1.60549e-01,  2.07368e-01,  1.77233e-01,  2.10677e-01,
       -2.96230e-02, -5.69800e-02, -3.56750e-02, -3.05980e-02,
       -4.95960e-02, -4.94250e-02,  7.94000e-04, -3.61090e-02,
       -4.71360e-02, -2.40330e-02, -3.86660e-02, -3.41220e-02,
       -8.22000e-03, -3.37920e-02, -4.25360e-02, -1.75520e-02,
       -2.64640e-02, -1.51640e-02, -5.93200e-03, -4.96030e-02,
       -7.39780e-02, -2.81280e-02, -8.57600e-03, -1.06940e-02,
        9.28700e-03, -5.85290e-02, -6.73740e-02, -2.62440e-02,
        1.28230e-02,  3.95600e-03,  2.36120e-02, -4.47240e-02,
       -8.01490e-02, -6.16540e-02,  7.29800e-03,  8.24100e-03,
        3.72860e-02, -4.43750e-02, -5.39120e-02, -6.89890e-02,
       -4.99170e-02, -6.63590e-02,  5.85640e-02, -2.34530e-02,
       -3.89350e-02, -7.50860e-02, -8.57970e-02, -9.50870e-02,
        3.69730e-02, -6.90160e-02, -3.18980e-02, -5.82830e-02,
       -8.24470e-02, -7.03910e-02, -3.74630e-02, -9.00970e-02,
       -1.31712e-01, -1.78080e-01, -2.04394e-01, -1.84273e-01])
initial_dvs['sweep'] =  np.array([0.070964]) * 10.0
initial_dvs['twist'] =  np.array([-0.026842,  0.005322,  0.002176,  0.019566,  0.022737,  0.039258,
       -0.011291])*100.0

initial_mission_vars = {}

num_routes = allocation_data['num']

initial_mission_vars[0, 'M0'] =  np.array([0.865])
initial_mission_vars[0, 'h_km_cp'] =  np.array([ 0.      , 10.352614,  9.639995,  9.806745,  9.741368,  9.754352,
        9.790615,  9.907972, 10.078964, 10.39693 , 10.715938, 11.128908,
       11.587087, 11.761091, 12.029878, 12.073079, 12.31011 , 12.980818,
        2.755118,  0.      ])
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
initial_mission_vars[2, 'M0'] =  np.array([0.865])
initial_mission_vars[2, 'h_km_cp'] =  np.array([ 0.      ,  3.358386, 11.794314, 10.570169, 11.222908, 11.137243,
       11.40959 , 11.508232, 11.644192, 11.766874, 11.852002, 12.016006,
       11.989308, 12.315591, 11.90018 , 12.966464, 10.912562,  4.543818,
        1.14372 ,  0.      ])
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
initial_mission_vars[5, 'M0'] =  np.array([0.861248])
initial_mission_vars[5, 'h_km_cp'] =  np.array([ 0.      ,  0.612359,  2.437157,  5.259522,  8.652742, 11.546152,
       13.      , 13.      , 13.      , 12.522471, 12.42453 , 12.185431,
       10.444275,  7.782173,  5.518857,  3.505517,  1.977432,  0.882418,
        0.214024,  0.      ])
initial_mission_vars[6, 'M0'] =  np.array([0.865])
initial_mission_vars[6, 'h_km_cp'] =  np.array([ 0.      ,  2.215355,  9.220815, 11.79736 , 11.479506, 11.738718,
       11.71313 , 11.828629, 11.878935, 11.96389 , 12.029362, 12.101546,
       12.161716, 12.227062, 12.249683, 12.636535,  7.175313,  3.187874,
        0.764711,  0.      ])
initial_mission_vars[7, 'M0'] =  np.array([0.865])
initial_mission_vars[7, 'h_km_cp'] =  np.array([ 0.      ,  2.414538, 10.050323, 11.546091, 11.465041, 11.652454,
       11.670448, 11.776194, 11.837535, 11.933405, 11.988602, 12.102384,
       12.093759, 12.312198, 12.036121, 13.      ,  7.88779 ,  3.409916,
        0.860561,  0.      ])
initial_mission_vars[8, 'M0'] =  np.array([0.865])
initial_mission_vars[8, 'h_km_cp'] =  np.array([ 0.      ,  1.411425,  5.633296, 10.629802, 12.188985, 11.846093,
       12.035228, 12.001881, 12.082329, 12.110118, 12.162384, 12.206445,
       12.238263, 12.307784, 12.485787,  8.163718,  4.594912,  2.008043,
        0.492263,  0.      ])
initial_mission_vars[9, 'M0'] =  np.array([0.864832])
initial_mission_vars[9, 'h_km_cp'] =  np.array([ 0.      ,  5.336129, 11.675169,  9.792977, 10.695585, 10.43858 ,
       10.818874, 10.901298, 11.211396, 11.45758 , 11.641887, 11.780121,
       11.951703, 12.010029, 12.239684, 12.042561, 12.999181,  7.034103,
        1.478328,  0.      ])

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


prob['revenue:x1'] = np.array([4.346   , 1.343   , 1.756   , 0.704982, 3.657   , 0.618803,
                               1.32    , 1.389   , 0.980978, 2.425   , 1.965   ]) * 1.0e3
prob['revenue:x2'] =  np.array([6.036388, 1.161386, 1.523723, 0.630445, 3.216191, 0.57318 ,
                                1.140746, 1.203193, 0.857275, 2.115696, 1.705637]) * 1.0e3
prob['revenue:y1'] =  np.array([2.203788, 0.447937, 0.585859, 0.237531, 1.219083, 0.205585,
                                0.441269, 0.463263, 0.32806 , 0.808068, 0.657302]) * 1.0e3
prob['revenue:y2'] =  np.array([3.103576, 0.649196, 0.851509, 0.332958, 1.753939, 0.29336 ,
                                0.637838, 0.672619, 0.47078 , 1.172821, 0.951504]) * 1.0e3
prob['revenue:z1'] =  np.array([0.3  , 3.192, 4.112, 2.242, 1.248, 0.3  , 0.716, 1.896, 2.264,
                                0.416, 0.416]) * 1.0e3

prob.run_model()

prob.model.allocation_mission_group.allocation_group.profit_comp.list_outputs()
prob.model.allocation_mission_group.revenue_comp.list_outputs()

for name in ['flt_day', 'revenue:x1', 'revenue:x2', 'revenue:y1', 'revenue:y2', 'revenue:z1', 'costs', 'revenue', 'tot_pax', 'pax_flt', 'nacc', 'p', 'profit']:
    print(name, prob[name])


print(prob.driver.get_design_var_values())