"""
Run Amiego plus AMD
"""
import os

import numpy as np

from six import iteritems

from openmdao.api import Problem, Group, IndepVarComp, ParallelGroup, NonlinearBlockJac, LinearBlockJac
from openmdao.parallel_api import PETScVector

from amd_om.mission_analysis.components.aerodynamics.rans_3d_data import get_aero_smt_model, get_rans_crm_wing
from amd_om.mission_analysis.components.propulsion.b777_engine_data import get_prop_smt_model
from amd_om.mission_analysis.mission_group import MissionGroup


allocation_data = {}
allocation_data['num_existing'] = 2
allocation_data['num_new'] = 1
allocation_data['existing_names'] = ['B738', 'B747']
allocation_data['new_names'] = ['CRM']
allocation_data['names'] = [ 'CRM', 'B738', 'B747']

allocation_data['num'] = 2

allocation_data['range_km'] = np.array([6928.15,    2145.99]) * 1.852

allocation_data['num_pt'] = 20 * np.ones(2, int)
allocation_data['num_cp'] = 5 * np.ones(2, int)


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
        self.linear_solver = LinearBlockJac()


class AllocationMissionGroup(Group):

    def initialize(self):
        self.metadata.declare('allocation_data', types=dict)

        self.metadata.declare('ref_area_m2', default=10., types=np.ScalarType)
        self.metadata.declare('Wac_1e6_N', default=1., types=np.ScalarType)
        self.metadata.declare('Mach_mode', 'TAS', values=['TAS', 'EAS', 'IAS', 'constant'])

        self.metadata.declare('propulsion_model')
        self.metadata.declare('aerodynamics_model')

        self.metadata.declare('initial_mission_vars', default=None, types=dict, allow_none=True)

    def setup(self):
        meta = self.metadata

        allocation_data = meta['allocation_data']

        ref_area_m2 = meta['ref_area_m2']
        Wac_1e6_N = meta['Wac_1e6_N']
        Mach_mode = meta['Mach_mode']

        propulsion_model = meta['propulsion_model']
        aerodynamics_model = meta['aerodynamics_model']

        initial_mission_vars = meta['initial_mission_vars']

        allocation_data = meta['allocation_data']

        multi_mission_group = MultiMissionGroup(
            general_allocation_data={'weight_pax_N' : 84*9.81},
            allocation_data=allocation_data,
            ref_area_m2=ref_area_m2, Wac_1e6_N=Wac_1e6_N, Mach_mode=Mach_mode,
            propulsion_model=propulsion_model, aerodynamics_model=aerodynamics_model,
            initial_mission_vars=initial_mission_vars,
        )

        self.add_subsystem('multi_mission_group', multi_mission_group, promotes=['*'])


#-------------------------------------------------------------------------


scaler = 7.841405
ref_area_m2 = 3.407014 * scaler ** 2 * 2
Wac_1e6_N = 0.1381 * 9.81
Mach_mode = 'TAS'

propulsion_model = get_prop_smt_model()
aerodynamics_model = get_aero_smt_model()

xt, yt, xlimits = get_rans_crm_wing()
aerodynamics_model.xt = xt


prob = Problem(model=AllocationMissionGroup(allocation_data=allocation_data,
                                            ref_area_m2=ref_area_m2, Wac_1e6_N=Wac_1e6_N, Mach_mode=Mach_mode,
                                            propulsion_model=propulsion_model, aerodynamics_model=aerodynamics_model))


prob.model.add_subsystem('p_CLt', IndepVarComp('CLt', np.zeros((36, ))), promotes=['*'])
prob.model.add_subsystem('p_CDt', IndepVarComp('CDt', np.zeros((36, ))), promotes=['*'])

prob.setup(vector_class=PETScVector)

prob['CLt'] = np.array([ 0.27891953,  0.27740675,  0.50567129,  0.50411111,  0.67178486,  0.67072796,
  0.20625611,  0.2048599,   0.45098623,  0.44937616,  0.65085738,  0.6492935,
  0.12143678,  0.11953893,  0.39299682,  0.39062201,  0.63840812,  0.63517673,
  0.12498458,  0.12133426,  0.38719886,  0.38262164,  0.63962429,  0.63352078,
  0.12739122,  0.12252601,  0.38887416,  0.38271344,  0.65587936,  0.64779034,
  0.13010178,  0.12376715,  0.40004596,  0.39172921,  0.67063932,  0.66241266])
prob['CDt'] = np.array([ 0.01245633,  0.01265379,  0.02219869,  0.02234761,  0.04310465,  0.04315868,
  0.01059043,  0.01074938,  0.01776348,  0.01788232,  0.03369645,  0.03376599,
  0.00980034,  0.00998945,  0.01497374,  0.01511459,  0.02877744,  0.02887994,
  0.00996479,  0.0103015,   0.01479722,  0.01504052,  0.02780155,  0.02791219,
  0.01017521,  0.01059275,  0.01503608,  0.01532366,  0.02923256,  0.02926247,
  0.01062982,  0.01110888,  0.01586912,  0.01614131,  0.03669549,  0.03639307])

prob.run_model()

print('CLt', prob['CLt'])
print('CDt', prob['CDt'])
prob.check_totals()



