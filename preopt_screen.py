"""
Subclassed pyOptsparse driver that pre-screens the point to determine if it is worth running
the optimization.
"""
from __future__ import print_function

from six import iteritems
from six.moves import range

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
            print("Optimizing this Point!")
            fail = super(pyOptSparseWithScreening, self).run()

            print("Fail Flag from Driver:", fail)

            if not fail:
                return False

            try:
                code = self.pyopt_solution.optInform['value']
            except:
                # Hard crash probably, so keep it as a failure.
                print("Couldn't key into pyopt_solution.")
                print("Fail Flag:", fail)
                return True

            # Call it a sucess when current point can't be improved.
            if code[0] == 41:
                "Pyoptsparse returned Code 41: current point cannot be improved."
            elif code[0] == 63:
                "Pyoptsparse returned Code 63: unable to proceed into undefined region."
            else:
                print("Pyoptsparse returned Code", code[0])
                return True

            # Now, one last desparate check. For codes 41 and 63, if they look good, we
            # keep them.

            sn_file = 'SNOPT_summary.out'

            with open(sn_file, 'r') as textfile:
                # The text of the entire sourcefile
                lines = textfile.readlines()

            obj = None

            # Read Objective
            for j, line in enumerate(lines):
                if "Objective value" in line:
                    _, obj = line.split("Objective value")
                    obj = obj.strip()
                    obj = float(obj)
                    break

            if obj is None:
                print("Error reading SNOPT file. Discarding point.")

            # Read feasibility from last convergence line
            line = lines[j-6]
            if line[30] == '(':
                print("Solution is feasible, keeping it.")
                print("Objective:", obj)
                return False
            else:
                print("Solution is not feasible. Discarding point.")


        else:

            # Poke approximate profit and constraint values.
            print("Skipping this Point!")
            print("profit", apx_profit)
            print("cons", apx_cons)
            self.success = False
            obj = list(self.get_objective_values().keys())[0]

            problem = self._problem
            problem.model._outputs[obj] = apx_profit

            for name, value in iteritems(apx_cons):
                try:
                    problem.model._outputs[name] = value

                # This can only happen under MPI when a constraint is only on a subset of procs.
                except KeyError:
                    pass

        return True


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
        model = self._problem.model
        preopt_flag = True

        # Run Model
        self.allocation_data['scale_fac'] = 0.95
        model._solve_nonlinear()
        self.allocation_data['scale_fac'] = 1.0

        prom2abs = model._var_allprocs_prom2abs_list['output']
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
