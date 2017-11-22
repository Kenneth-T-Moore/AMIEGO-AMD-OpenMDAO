"""
Subclassed pyOptsparse driver that pre-screens the point to determine if it is worth running
the optimization.
"""
from __future__ import print_function

from six import iteritems

import numpy as np
from mpi4py import MPI

from openmdao.api import pyOptSparseDriver


class pyOptSparseWithScreening(pyOptSparseDriver):

    def run(self):
        """
        Excute pyOptsparse with pre-screening.

        Note that pyOpt controls the execution, and the individual optimizers
        (e.g., SNOPT) control the iteration.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        run_flag, apx_profit, apx_cons = self.preopt_screen()

        if run_flag:

            # Do continuous optimization
            super(pyOptSparseWithScreening, self).run()

            try:
                code = self.pyopt_solution.optInform['value']
            except:
                # Hard crash probably, so keep it as a failure.
                return

            # Call it a sucess when current point can't be improved.
            if code[0] == 41:
                self.success = True

            cons = self.get_constraint_values()
            tol = self.opt.getOption('Major feasibility tolerance')
            tol_opt = self.opt.getOption('Major optimality tolerance')

            print(code[0])
            print(self.success)

            # If solution is feasible we proceed with it
            con_meta = self._cons
            feasible = True
            for name, meta in iteritems(con_meta):
                val = cons[name]
                upper = meta['upper']
                lower = meta['lower']
                equals = meta['equals']

                if upper is not None and any(val > upper + tol):
                    feasible = False
                    break
                if lower is not None and any(val < lower - tol):
                    feasible = False
                    break
                if equals is not None and any(abs(val - equals) > tol):
                    feasible = False
                    break

            if feasible:

                # Soln is feasible; obj and cons already in openmdao.
                self.success = True
                return

        else:

            # Poke approximate profit and constraint values.
            print("Failed Pre-Opt!")
            print("profit", apx_profit)
            print("cons", apx_cons)
            self.success = False
            obj = list(self.get_objective_values().keys())[0]

            problem = self._problem
            problem.model._outputs[obj] = apx_profit

            for name, value in iteritems(apx_cons):
                # problem.model._outputs[name] = value[2:]
                problem.model._outputs[name] = value

    def preopt_screen(self):
        """
        Pre-screen the integer inputs to determine if the continuous optimization should be run.

        Returns
        -------
        bool
            True if continuous optimization should run
        float
            Approximate profit value to use in place of continuous if not run.
        ndarray
            Approzimate array of constraint values to use in place of continuous
            run if not run.
        """
        preopt_flag = True

        # Run Model
        self.allocation_data['scale_fac'] = 0.95
        model._solve_nonlinear()
        self.allocation_data['scale_fac'] = 1.0

        prom2abs = self._problem.model._var_allprocs_prom2abs_list['output']
        profit_key = prom2abs['profit'][0]
        conkey1 = prom2abs['g_aircraft_new'][0]
        conkey2 = prom2abs['g_aircraft_exist'][0]

        profit = self.get_objective_values()[profit_key]
        cons = self.get_constraint_values()

        for key in [conkey1, conkey2]:
            con = cons[key]
            for kk in range(len(con)):
                if con[kk] > (1.0 + 1.0e-6): #New aircraft check within 1%, hoping optimizer can make it feasible
                    preopt_flag = False

        return preopt_flag, profit/-1.0e3, cons
