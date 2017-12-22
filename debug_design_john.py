"""
Example of an ADFLOW problem that gave negative volumes and locked up on writing the failed meshes.
"""
import os

from six import iteritems

import numpy as np

from openmdao.api import Problem, Group, Driver
from openmdao.parallel_api import PETScVector

from amd_om.design.design_group import DesignGroup
from amd_om.design.utils.flight_conditions import get_flight_conditions
from amd_om.utils.aircraft_data.CRM_full_scale import get_aircraft_data

from amd_om.utils.pre_setup import aeroOptions, meshOptions


class MyDriver(Driver):

    def run(self):
        print("Design Vars (Scaled)")
        dvs = self.get_design_var_values()
        for name, value in iteritems(dvs):
            print(name, value)

        super(MyDriver, self).__init__()


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
initial_dvs['shape'] = np.array([ -4.66283369e-02,   1.51054035e-01,  -2.84069976e-02,
         3.56930352e-02,  -7.27332793e-02,  -2.13784323e-01,
         1.37011139e-01,  -7.28824974e-02,  -7.72100750e-02,
        -1.39994481e-01,  -1.28785427e-01,   4.73959392e-02,
         9.88813318e-02,   5.13524381e-02,   3.29567365e-03,
        -4.88518910e-02,  -9.16143213e-02,  -1.18139370e-01,
         8.42163873e-02,   3.41767015e-02,   3.47273680e-02,
         5.12647978e-02,   5.05334627e-02,   2.39721305e-02,
        -3.62280645e-03,  -1.61773233e-02,   3.83128649e-03,
         2.26354974e-01,   2.50550750e-02,   5.51677570e-02,
        -4.85150021e-02,  -1.28536854e-01,  -1.27480534e-01,
        -1.22242634e-01,   1.39981123e-02,   3.62632141e-02,
         3.90368564e-02,   6.19866867e-02,   1.05521452e-01,
         1.62146101e-01,   1.03928386e-01,  -2.06775772e-01,
        -5.05165160e-02,   3.94402047e-01,   2.45702540e-01,
         2.25780066e-02,  -9.05002766e-02,  -3.66562094e-03,
         5.15785734e-02,   1.00365823e-01,  -3.69250618e-02,
        -1.66729609e-02,  -5.72916682e-03,   1.47225184e-02,
         2.50675390e-02,  -3.67835584e-02,   4.68969480e-02,
         1.46912558e-02,  -1.01028709e-01,  -1.34163279e-01,
        -3.28565683e-02,  -1.35498420e-01,  -1.33078464e-01,
        -1.49128005e-01,  -1.74238580e-01,  -2.16166967e-01,
        -1.80260722e-01,  -6.28373348e-02,  -8.10327630e-02,
        -6.77884212e-02,  -7.95330358e-02,  -6.14262993e-02,
        -1.30828854e-02,   2.53659475e-02,   5.30074674e-02,
         6.35955420e-02,   6.65488121e-02,   4.30813999e-02,
        -4.00229439e-02,   1.81260211e-02,   5.02087281e-02,
         6.25082191e-02,   5.96106652e-02,   4.32690475e-02,
        -4.62760776e-02,   7.57512051e-02,   9.18224431e-02,
         8.44688267e-02,   4.49809649e-02,   4.33319172e-02,
         8.50853298e-02,   1.43221181e-01,   1.32873251e-01,
         1.05816254e-01,   1.06633001e-01,   9.36676548e-02,
         8.97123776e-02,   1.53820559e-01,   1.35822913e-01,
         1.12314813e-01,   1.11171936e-01,   4.52185609e-02,
         6.53775567e-02,   1.07990835e-01,   8.94558023e-02,
         2.20448919e-02,   4.14925506e-02,   3.47323582e-02,
        -6.48413458e-03,   2.80787213e-02,  -1.20342317e-02,
        -1.02244874e-03,   3.36158942e-02,  -2.54416443e-04,
        -6.42685795e-02,  -1.64769804e-02,  -2.62735498e-03,
         3.35368425e-02,   1.14534803e-02,  -1.65533273e-02,
        -1.17694591e-01,  -9.11179797e-02,  -5.83496813e-02,
        -1.98676114e-02,   6.86940576e-03,  -5.47264851e-02,
        -5.87883785e-02,   6.19401716e-02,   1.38505038e-01,
         1.89005465e-01,   1.71667810e-01,   2.21336435e-01,
        -1.62315514e-01,  -1.72967467e-01,  -4.78708104e-02,
         3.20061342e-03,  -6.75000317e-02,  -5.43100292e-02,
        -3.57170717e-01,  -2.50922338e-01,  -1.81285094e-01,
        -1.20664497e-01,  -1.01098613e-01,  -4.08949644e-02,
         1.33796820e-02,   1.03957456e-01,   5.42367860e-02,
        -1.74091810e-02,  -3.05489947e-02,  -1.96022831e-02,
         4.50998000e-01,   3.13734677e-01,   1.96379968e-01,
         1.33723076e-01,   1.11857669e-01,   1.04367666e-01,
         2.10960837e-01,   1.38551094e-01,   1.21778356e-01,
         1.35226106e-01,   1.17061534e-01,  -2.40916833e-02,
        -5.34277469e-02,  -1.33682901e-01,  -2.02618260e-01,
        -2.04158205e-01,  -2.01518639e-01,  -2.53463794e-01,
        -8.71041136e-02,  -2.12773173e-01,  -2.74920671e-01,
        -2.29716859e-01,  -1.23960506e-01,  -2.65534296e-02,
         4.47596938e-02,  -4.92881261e-04,   4.79960339e-02,
         1.77080639e-02,   6.28803853e-02,  -4.30141968e-03,
         8.66406987e-02,   1.19153388e-03,   2.88268397e-02,
         1.50747941e-02,  -7.17476522e-02,  -8.90623399e-02,
        -2.73361533e-02,  -4.97076774e-02,  -7.98437531e-02,
        -1.23893651e-01,  -1.87031379e-01,  -1.38126754e-01])
initial_dvs['twist'] = np.array([ 0.08      ,  0.08      ,  0.08      ,  0.06029069, -0.08      , -0.08      , -0.08      ])*100.0
initial_dvs['sweep'] = np.array([ 0.06380759])*10.0
initial_dvs['area'] = np.array([ 0.10086618])*10.0

prob = Problem()
prob.driver = MyDriver()
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