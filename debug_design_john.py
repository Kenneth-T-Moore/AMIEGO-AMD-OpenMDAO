"""
Example of an ADFLOW problem that gave negative volumes and locked up on writing the failed meshes.
"""
import os

from six import iteritems

import numpy as np

from openmdao.api import Problem, Group
from openmdao.parallel_api import PETScVector

from amd_om.design.design_group import DesignGroup
from amd_om.design.utils.flight_conditions import get_flight_conditions
from amd_om.utils.aircraft_data.CRM_full_scale import get_aircraft_data

from amd_om.utils.pre_setup import aeroOptions, meshOptions


flight_conditions = get_flight_conditions()
aircraft_data = get_aircraft_data()

aeroOptions = {'gridFile' : '../Plugins/amd_om/grids/L3_myscaled.cgns',
               'writesurfacesolution' : False,
               'writevolumesolution' : True,
               'writetecplotsurfacesolution' : False,
               'grad_scaler' : 10.,
               'outputDirectory' : './'
               }
meshOptions = {'gridFile' : '../Plugins/amd_om/grids/L3_myscaled.cgns'}

design_variables = ['shape', 'twist', 'sweep', 'area']

initial_dvs = {}
initial_dvs['shape'] = np.array([-0.06015097,  0.03539321,  0.04366222,  0.086622  ,  0.04269916,
                                 -0.08392644,  0.11244178, -0.12991661, -0.09072263, -0.13614747,
                                 -0.09281048,  0.04321413,  0.10096387,  0.06636394,  0.01911703,
                                 -0.03785342, -0.0625496 , -0.12099273,  0.02400916,  0.03176889,
                                 0.04074797,  0.0672233 ,  0.06485745,  0.03343983,  0.00195302,
                                 -0.01104444,  0.00368061,  0.19257507,  0.00111015,  0.03291845,
                                 0.05050331,  0.01568764,  0.00186865, -0.00446012,  0.08112198,
                                 0.07563683,  0.10236726,  0.16069289,  0.13970302,  0.1648042 ,
                                 0.0450799 ,  0.08913052,  0.11076426,  0.07223092,  0.03712591,
                                 -0.00168609, -0.01291175,  0.01209249,  0.01548182,  0.05675416,
                                 -0.03177877, -0.01138458,  0.007148  ,  0.00796317,  0.03180419,
                                 0.03458523,  0.02640677, -0.00984506, -0.04384874, -0.1982936 ,
                                 -0.09031243, -0.14002396, -0.09246067, -0.10202259, -0.15089192,
                                 -0.14151222, -0.12270829, -0.0391938 , -0.05150488, -0.08305141,
                                 -0.10465563, -0.10175164, -0.02512117,  0.01252088,  0.02668812,
                                 0.02272073,  0.02744459,  0.01167219, -0.01677699,  0.0380104 ,
                                 0.05307728,  0.04645567,  0.04731079,  0.04031935,  0.01402579,
                                 0.0760989 ,  0.07577983,  0.0700565 ,  0.07500427,  0.06345949,
                                 0.06002949,  0.10789629,  0.1022637 ,  0.08924496,  0.08974874,
                                 0.07555983,  0.08212461,  0.13668795,  0.11010598,  0.08543024,
                                 0.09244986,  0.0766276 ,  0.08048089,  0.11795764,  0.09593738,
                                 0.07349904,  0.07599661,  0.05731978,  0.01000462,  0.04374565,
                                 0.04097951,  0.03474099,  0.04901941,  0.01112648, -0.06111217,
                                 -0.02817478, -0.01461737,  0.01137851,  0.01800756, -0.02137717,
                                 -0.09948539, -0.08832951, -0.04982677, -0.02419183, -0.0145354 ,
                                 -0.02353688, -0.06833324,  0.08211106,  0.15087517,  0.19995204,
                                 0.17419371,  0.20602057, -0.02499472, -0.05948012, -0.03761644,
                                 -0.03676678, -0.05380523, -0.04827464, -0.00760588, -0.03258232,
                                 -0.03713605, -0.01900795, -0.03519369, -0.03652005, -0.00510485,
                                 -0.03490756, -0.05613908, -0.02134946, -0.0283795 , -0.01024956,
                                 -0.00633639, -0.05485978, -0.06471628, -0.02213878, -0.01286367,
                                 -0.0069117 ,  0.00795812, -0.0567236 , -0.07801191, -0.0448521 ,
                                 0.01270876, -0.00312514,  0.01976428, -0.0444744 , -0.08647099,
                                 -0.0532212 , -0.00094589, -0.01289192,  0.04175217, -0.03967634,
                                 -0.05346984, -0.06863361, -0.04416314, -0.05954628,  0.05622069,
                                 -0.01816169, -0.02268927, -0.06129057, -0.07307046, -0.08305317,
                                 0.04036049, -0.05757029, -0.02041531, -0.05356558, -0.07870195,
                                 -0.06207241, -0.03330645, -0.0892925 , -0.13763397, -0.18624647,
                                 -0.21577592, -0.19048606])
initial_dvs['twist'] = np.array([-0.02670733,  0.00396999,  0.00206548,  0.02087249,  0.02193591, 0.03856382, -0.02119188])
initial_dvs['sweep'] = np.array([ 0.09994278])
initial_dvs['area'] = np.array([ 0.10856287])

prob = Problem()
model = prob.model = Group()

design_group = DesignGroup(
    flight_conditions=flight_conditions, aircraft_data=aircraft_data,
    aeroOptions=aeroOptions, meshOptions=meshOptions, design_variables=design_variables,
)

model.add_subsystem('design_group', design_group, promotes=['*'])

prob.setup(vector_class=PETScVector)

for key, value in iteritems(initial_dvs):
    prob[key] = value

prob.run_driver()

print("Execution Completed!")