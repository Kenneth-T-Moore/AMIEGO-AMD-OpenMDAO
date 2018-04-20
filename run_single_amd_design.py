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
initial_dvs['area'] =  np.array([0.108597]) * 10.0
initial_dvs['shape'] =  np.array([-4.70340e-02,  7.43560e-02,  6.26410e-02,  9.51220e-02,
                           5.59360e-02, -7.26530e-02,  7.37180e-02, -1.50831e-01,
                           -7.86160e-02, -1.40746e-01, -1.28062e-01, -1.74740e-02,
                           2.93040e-02, -1.30860e-02, -5.29100e-02, -1.02336e-01,
                           -9.20490e-02, -1.19837e-01,  2.76210e-02,  3.15150e-02,
                           3.62160e-02,  5.17170e-02,  5.08490e-02,  2.42990e-02,
                           -4.15200e-03, -1.68880e-02,  8.19000e-04,  1.86303e-01,
                           2.16420e-02,  4.95800e-02,  7.52050e-02,  6.92950e-02,
                           5.92380e-02,  4.72800e-02,  7.90590e-02,  4.11780e-02,
                           4.87480e-02,  7.41000e-02,  4.36940e-02,  8.37150e-02,
                           9.63700e-02,  1.27261e-01,  1.32078e-01,  8.73990e-02,
                           4.59430e-02,  1.77650e-02,  1.84520e-02,  6.14690e-02,
                           4.81090e-02,  9.57450e-02, -3.88550e-02, -1.81100e-02,
                           -6.79000e-04,  1.55100e-02,  2.48900e-02,  3.80390e-02,
                           4.37860e-02,  1.29550e-02, -3.30160e-02, -1.63267e-01,
                           -3.85670e-02, -1.40316e-01, -1.35535e-01, -1.50316e-01,
                           -1.75419e-01, -1.34761e-01, -1.02670e-01, -5.75440e-02,
                           -7.29980e-02, -1.08691e-01, -1.39270e-01, -1.52795e-01,
                           -1.56530e-02,  2.30640e-02,  5.02620e-02,  5.86020e-02,
                           6.18580e-02,  3.93650e-02, -3.91440e-02,  1.96730e-02,
                           5.13910e-02,  6.33050e-02,  6.04940e-02,  4.29840e-02,
                           -4.37020e-02,  2.57610e-02,  3.82510e-02,  3.78580e-02,
                           4.92700e-02,  4.62590e-02, -1.56860e-02,  3.97970e-02,
                           4.11410e-02,  3.50590e-02,  4.69240e-02,  4.18010e-02,
                           -2.68100e-03,  5.91670e-02,  4.63560e-02,  3.50280e-02,
                           4.76280e-02,  4.64530e-02, -9.67800e-03,  4.10340e-02,
                           3.20890e-02,  2.46030e-02,  4.34050e-02,  3.57550e-02,
                           -7.34370e-02, -2.02380e-02, -1.07390e-02, -6.00000e-04,
                           3.30790e-02, -9.85000e-04, -1.21752e-01, -6.63020e-02,
                           -5.32600e-02, -1.66570e-02,  1.00750e-02, -1.82320e-02,
                           -1.19428e-01, -9.37550e-02, -6.18270e-02, -2.43050e-02,
                           2.00900e-03,  6.21300e-03, -5.92120e-02,  6.42490e-02,
                           1.41368e-01,  1.91938e-01,  1.74659e-01,  2.24442e-01,
                           1.83500e-03, -6.59400e-02, -4.88020e-02, -4.70370e-02,
                           -6.68730e-02, -5.51780e-02,  9.28800e-03, -1.36660e-02,
                           -3.27290e-02, -3.19860e-02, -3.83500e-02, -3.80050e-02,
                           2.94250e-02, -2.03000e-04, -1.52560e-02, -3.73000e-03,
                           -1.91940e-02, -1.05840e-02,  3.16480e-02, -5.18000e-03,
                           -2.96630e-02, -1.81460e-02, -1.79980e-02, -1.85870e-02,
                           4.92810e-02, -6.86900e-03, -3.45990e-02, -1.76640e-02,
                           -9.06600e-03, -2.28630e-02,  6.47890e-02,  1.17960e-02,
                           -2.43670e-02, -2.01760e-02, -2.48700e-03, -3.40700e-02,
                           8.04170e-02,  2.29730e-02,  3.31600e-03, -1.77280e-02,
                           2.82200e-03, -3.42660e-02,  1.18111e-01,  5.14780e-02,
                           3.81300e-02,  8.96000e-03,  3.69000e-03, -8.63500e-03,
                           8.11820e-02, -6.15000e-03,  1.97970e-02,  5.88000e-03,
                           -1.26290e-02, -9.17700e-03,  1.89410e-02, -5.12780e-02,
                           -8.25450e-02, -1.27373e-01, -1.60407e-01, -1.56030e-01])
initial_dvs['sweep'] =  np.array([-0.092868]) * 10.0
initial_dvs['twist'] =  np.array([-0.031206,  0.004141, -0.005796,  0.000981, -0.000921,  0.0169  ,
                           -0.035491]) * 10.0

initial_mission_vars = {}

num_routes = allocation_data['num']

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
recorder_file_name = 'recorder_amd.db'


print("Running Setup")
prob.setup(vector_class=PETScVector)
print("Setup Complete")

# Insert Optimal Design here

prob['flt_day'] = np.array([[ 0.,  0.,  0.,  0.,  4.,  6.,  3.,  0.,  7.,  0.,  2.],
                            [ 5.,  0.,  0.,  3.,  0.,  1.,  0.,  1.,  2.,  0.,  0.],
                            [ 5.,  2.,  1.,  8.,  2.,  0.,  0.,  0.,  1.,  0.,  0.]]).T


prob['revenue:x1'] = np.array([4.346   , 1.343   , 1.756   , 0.706155, 3.657   , 0.618899,
                                                           1.32    , 1.389   , 0.981   , 1.      , 1.965   ]) * 1.0e3
prob['revenue:x2'] =  np.array([6.018   , 1.159146, 1.509376, 0.630516, 3.200992, 0.697066,
                                                           1.148846, 1.202317, 0.854041, 1.      , 1.7004  ]) * 1.0e3
prob['revenue:y1'] =  np.array([2.6318  , 0.447763, 0.586488, 0.235803, 1.21878 , 0.210106,
                                                           0.441207, 0.463212, 0.327289, 0.809614, 0.674075]) * 1.0e3
prob['revenue:y2'] =  np.array([3.7277  , 0.648142, 0.844902, 0.333115, 1.746825, 0.334863,
                                                           0.641266, 0.672299, 0.470384, 1.      , 0.949216]) * 1.0e3
prob['revenue:z1'] =  np.array([0.3  , 3.192, 4.112, 2.242, 1.248, 0.3  , 0.716, 1.896, 2.264, 0.   , 0.416]) * 1.0e3

prob.run_model()

prob.model.allocation_mission_group.allocation_group.profit_comp.list_outputs()
prob.model.allocation_mission_group.revenue_comp.list_outputs()

for name in ['flt_day', 'revenue:x1', 'revenue:x2', 'revenue:y1', 'revenue:y2', 'revenue:z1', 'costs', 'revenue', 'tot_pax', 'pax_flt', 'nacc', 'p', 'profit']:
    print(name, prob[name])

prob.run_model()

for name in ['flt_day', 'revenue:x1', 'revenue:x2', 'revenue:y1', 'revenue:y2', 'revenue:z1', 'costs', 'revenue', 'tot_pax', 'pax_flt', 'nacc', 'p', 'profit']:
    print(name, prob[name])

