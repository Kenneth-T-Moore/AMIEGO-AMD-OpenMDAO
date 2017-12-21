"""
Run Amiego plus AMD
"""
import pickle
import os

import numpy as np

from six import iteritems

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.parallel_api import PETScVector
from openmdao.utils.mpi import MPI

from amd_om.design.utils.flight_conditions import get_flight_conditions
from amd_om.mission_analysis.components.aerodynamics.rans_3d_data import get_aero_smt_model, get_rans_crm_wing
from amd_om.mission_analysis.components.propulsion.b777_engine_data import get_prop_smt_model
from amd_om.utils.aircraft_data.CRM_full_scale import get_aircraft_data

from economy import Profit, RevenueManager
from prob_11_2_updated import allocation_data
from prob_11_2_general_allocation import general_allocation_data


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

        allocation_group = AllocationGroup(
            general_allocation_data=general_allocation_data, allocation_data=allocation_data,
        )

        self.add_subsystem('inputs_comp', inputs_comp, promotes=['*'])
        self.add_subsystem('revenue_comp', revenue_comp, promotes=['*'])
        self.add_subsystem('allocation_group', allocation_group, promotes=['*'])

        demand_constraint = np.zeros((num_routes, num_aircraft))
        for ind_ac in range(num_aircraft):
            aircraft_name = allocation_data['names'][ind_ac]
            for ind_rt in range(num_routes):
                demand_constraint[ind_rt, ind_ac] = allocation_data['capacity', aircraft_name]

        self.add_design_var('flt_day', lower=flt_day_lower, upper=flt_day_upper)
        self.add_design_var('revenue:x1', lower=0.0, ref=1.0e4)
        self.add_design_var('revenue:y1', lower=0.0, ref=1.0e4)
        self.add_design_var('revenue:x2', lower=0.0, ref=1.0e4)
        self.add_design_var('revenue:y2', lower=0.0, ref=1.0e4)
        self.add_design_var('revenue:z1', lower=0.0, ref=1.0e4)
        self.add_objective('profit')


class AllocationMissionDesignGroup(Group):

    def initialize(self):
        self.metadata.declare('flight_conditions', types=dict)
        self.metadata.declare('aircraft_data', types=dict)

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

record = True

initial_dvs = {}


initial_mission_vars = {}

num_routes = allocation_data['num']
aircraft_data = get_aircraft_data()

ref_area_m2 = aircraft_data['areaRef_m2']
Wac_1e6_N = aircraft_data['Wac_1e6_N']
Mach_mode = 'TAS'

propulsion_model = get_prop_smt_model()
aerodynamics_model = get_aero_smt_model()

xt, yt, xlimits = get_rans_crm_wing()
aerodynamics_model.xt = xt


prob = Problem(model=AllocationMissionDesignGroup(flight_conditions=flight_conditions, aircraft_data=aircraft_data,
                                                  general_allocation_data=general_allocation_data, allocation_data=allocation_data,
                                                  ref_area_m2=ref_area_m2, Wac_1e6_N=Wac_1e6_N, Mach_mode=Mach_mode,
                                                  propulsion_model=propulsion_model, aerodynamics_model=aerodynamics_model,
                                                  initial_mission_vars=initial_mission_vars))


print("Running Setup")
prob.setup(vector_class=PETScVector)
print("Setup Complete")
for key, value in iteritems(initial_dvs):
    prob[key] = value

prob.run_model()
derivs = prob.compute_totals(of=['profit'], wrt=['revenue:x1', 'revenue:x2', 'revenue:y1', 'revenue:y2', 'revenue:z1'])
print(derivs)

print('done')


