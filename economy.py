"""
Revenue Management System for AMD optimization.
"""
from __future__ import print_function

from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent


# Attractiveness coefficients for fare classes
a = np.array([0.864, -0.038])
b = np.array([-0.02, 0.016])
c = np.array([0.009, 0.008])


def calc_revenue(x1, y1, x2, y2, z1, alpha, beta, Cap):

    x = np.array([x1, x2])
    y = np.array([y1, y2])

    ntot = alpha - beta*y
    p = 1.0/(1.0 + np.exp(a - b*y + c*x))

    nacc = np.zeros((2, ))
    R = np.zeros((2, ))

    nacc[0] = min(ntot[0], z1)
    R[0] = (p[0]*x[0] + (1.0 - p[0])*y[0])*nacc[0]

    nacc[1] = min(ntot[1], (Cap - nacc[0]))
    R[1] = (p[1]*x[1] + (1.0 - p[1])*y[1])*nacc[1]

    Rev = np.sum(R)
    sum_nacc = np.sum(nacc)

    return Rev, sum_nacc, nacc, p, ntot


def demand_bounds(dem, price, range_route):
    LF1 = 0.6; LF2 = 1.0-LF1
    k=1.2

    p2 = price*k
    p1 = price*(1.0 - k*LF2)/LF1

    alp1, bet1 = price_elasticity(LF1, dem, p1, range_route)
    alp2, bet2 = price_elasticity(LF2, dem, p2, range_route)

    alpha = np.array([[alp1,alp2]])
    beta = np.array([[bet1,bet2]])
    return alpha, beta


def price_elasticity(LF, avail_seat, price, ran):
    alt_mode = 1 #Set this to 1 if alternate mode of transportation is available, otherwise 0

    dem = LF*avail_seat
    RVector_km = ran
    if alt_mode == 1:
        if (RVector_km < 300.0):
            range_elas = -1.6141
        elif (RVector_km < 1300.0):
            range_elas = -9.084E-07*RVector_km**2 + 2.319E-03*RVector_km - 2.228E+00
        else:
            range_elas =  -0.7485
    else:
        if (RVector_km < 300.0):
            range_elas = -1.2360
        elif (RVector_km < 900.0):
            range_elas = -1.614E-06*RVector_km**2 + 2.761E-03*RVector_km - 1.919E+00
        else:
            range_elas =  -0.7392

    alp = dem*(1.0-range_elas)
    bet = -range_elas*(dem/price)
    return alp, bet


class RevenueManager(ExplicitComponent):
    """
    Uses revenue model to output passengers demand per route.
    """
    def initialize(self):
        self.metadata.declare('general_allocation_data', type_=dict)
        self.metadata.declare('allocation_data', type_=dict)

    def setup(self):
        allocation_data = self.metadata['allocation_data']
        num_routes = allocation_data['num']
        num_existing_aircraft = allocation_data['num_existing']
        num_new_aircraft = allocation_data['num_new']
        num_aircraft = num_existing_aircraft + num_new_aircraft

        self.add_input('revenue:x1', shape=(num_routes, ))
        self.add_input('revenue:y1', shape=(num_routes, ))
        self.add_input('revenue:x2', shape=(num_routes, ))
        self.add_input('revenue:y2', shape=(num_routes, ))
        self.add_input('revenue:z1', shape=(num_routes, ))

        self.add_input('flt_day', shape=(num_routes, num_aircraft))

        self.add_output('pax_flt', shape=(num_routes, num_aircraft))
        self.add_output('revenue', shape=(num_routes, ))
        self.add_output('tot_pax', shape=(num_routes, ))

        # These are used in the final comparison spreadsheet.
        self.add_output('nacc', shape=(num_routes, 2))
        self.add_output('p', shape=(num_routes, 2))

        # Derivs of everything wrt just the continuous vars.
        self.declare_partials(of='*', wrt='revenue*')

        # Calculate Demand bounds for each route.
        self.alpha = np.empty((num_routes, 2))
        self.beta = np.empty((num_routes, 2))
        demand = allocation_data['demand']
        price = allocation_data['price_pax', 'CRM']
        route = allocation_data['range_km']
        for jj in range(num_routes):
            alp, bet = demand_bounds(demand[jj], price[jj], route[jj])
            self.alpha[jj, :] = alp
            self.beta[jj, :] = bet

    def compute(self, inputs, outputs):
        allocation_data = self.metadata['allocation_data']
        num_routes = allocation_data['num']
        num_existing_aircraft = allocation_data['num_existing']
        num_new_aircraft = allocation_data['num_new']
        num_ac = num_existing_aircraft + num_new_aircraft

        seats = []
        for key in allocation_data['names']:
            seats.append(allocation_data['capacity', key])
        seats = np.array(seats)

        trip = inputs['flt_day']
        x1 = inputs['revenue:x1']
        y1 = inputs['revenue:y1']
        x2 = inputs['revenue:x2']
        y2 = inputs['revenue:y2']
        z1 = inputs['revenue:z1']
        alpha = self.alpha
        beta = self.beta

        # Total number of seats available per day for a route.
        max_avail_seats = seats.dot(trip.T)

        # Main Revenue calculation.
        rev = np.zeros((num_routes, ))
        pax = np.zeros((num_routes, num_ac))
        test = np.zeros((num_routes, 2))
        tot_pax = np.empty((num_routes, ))
        for jj in range(num_routes):
            if max_avail_seats[jj] > 0:
                rev_j, totnacc_j, nacc, p, _ = calc_revenue(x1[jj], y1[jj], x2[jj], y2[jj], z1[jj],
                                                         alpha[jj, :], beta[jj, :],
                                                         max_avail_seats[jj])

                outputs['nacc'][jj, :] = nacc
                outputs['p'][jj, :] = p

                rev[jj] = rev_j
                for kk in range(num_ac):
                    x_kj = trip[jj, kk]
                    if x_kj > 0:
                        pax[jj, kk] = min(seats[kk], seats[kk]*totnacc_j/max_avail_seats[jj])
            else:
                totnacc_j = 0.0

            tot_pax[jj] = totnacc_j

        outputs['pax_flt'] = pax
        outputs['revenue'] = rev
        outputs['tot_pax'] = tot_pax

    def compute_partials(self, inputs, partials):
        allocation_data = self.metadata['allocation_data']
        num_routes = allocation_data['num']
        num_existing_aircraft = allocation_data['num_existing']
        num_new_aircraft = allocation_data['num_new']
        num_ac = num_existing_aircraft + num_new_aircraft

        seats = []
        for key in allocation_data['names']:
            seats.append(allocation_data['capacity', key])
        seats = np.array(seats)

        trip = inputs['flt_day']
        x1 = inputs['revenue:x1']
        y1 = inputs['revenue:y1']
        x2 = inputs['revenue:x2']
        y2 = inputs['revenue:y2']
        z1 = inputs['revenue:z1']

        max_avail_seats = seats.dot(trip.T)
        alpha = self.alpha
        beta = self.beta

        for jj in range(num_routes):
            Cap = max_avail_seats[jj]
            if Cap == 0:
                continue

            # Combined for ease
            x = np.array([x1[jj], x2[jj]])
            y = np.array([y1[jj], y2[jj]])

            ntot = alpha[jj, :] - beta[jj, :]*y
            dntot_dy = -beta[jj, :]

            p = 1.0/(1.0 + np.exp(a - b*y + c*x))
            fact = -np.exp(a - b*y + c*x)*p**2
            dp_dy = -fact*b
            dp_dx = fact*c

            if ntot[0] > z1[jj]:
                nacc0 = z1[jj]
                dnacc0_dy = 0.0
                dnacc0_dz = 1.0
            else:
                nacc0 = ntot[0]
                dnacc0_dy = dntot_dy[0]
                dnacc0_dz = 0.0

            dR0_dx = nacc0 * (p[0] + (x[0] - y[0])*dp_dx[0])
            dR0_dy = nacc0 * ((1.0 - p[0]) + (x[0] - y[0])*dp_dy[0]) + \
                    (p[0]*x[0] + (1.0 - p[0])*y[0])*dnacc0_dy
            dR0_dz = (p[0]*x[0] + (1.0 - p[0])*y[0])*dnacc0_dz

            if ntot[1] > (Cap - nacc0):
                nacc1 = (Cap - nacc0)
                dnacc1_dy = -dnacc0_dy
                dnacc1_dz = -dnacc0_dz
            else:
                nacc1 = ntot[1]
                dnacc1_dy = dntot_dy[1]
                dnacc1_dz = 0.0

            dR1_dx = nacc1 * (p[1] + (x[1] - y[1])*dp_dx[1])
            dR1_dy = nacc1 * ((1.0 - p[1]) + (x[1] - y[1])*dp_dy[1]) + \
                    (p[1]*x[1] + (1.0 - p[1])*y[1])*dnacc1_dy
            dR1_dz = (p[1]*x[1] + (1.0 - p[1])*y[1])*dnacc1_dz

            partials['revenue', 'revenue:x1'][jj][jj] = dR0_dx
            partials['revenue', 'revenue:x2'][jj][jj] = dR1_dx
            partials['revenue', 'revenue:y1'][jj][jj] = dR0_dy
            partials['revenue', 'revenue:y2'][jj][jj] = dR1_dy
            partials['revenue', 'revenue:z1'][jj][jj] = dR0_dz + dR1_dz

            sum_nacc = nacc0 + nacc1
            partials['tot_pax', 'revenue:y1'][jj][jj] = dnacc0_dy
            partials['tot_pax', 'revenue:y2'][jj][jj] = dnacc1_dy
            partials['tot_pax', 'revenue:z1'][jj][jj] = dnacc0_dz + dnacc1_dz

            for kk in range(num_ac):
                x_kj = trip[jj, kk]
                if x_kj > 0.0 and sum_nacc < max_avail_seats[jj]:
                    fact = seats[kk] / max_avail_seats[jj]
                    ii = jj * num_ac + kk
                    partials['pax_flt', 'revenue:y1'][ii, jj] = dnacc0_dy * fact
                    partials['pax_flt', 'revenue:y2'][ii, jj] = dnacc1_dy * fact
                    partials['pax_flt', 'revenue:z1'][ii, jj] = (dnacc0_dz + dnacc1_dz) * fact


class Profit(ExplicitComponent):
    """
    Calculate values for airline profit and constraints.
    """
    def initialize(self):
        self.metadata.declare('general_allocation_data', type_=dict)
        self.metadata.declare('allocation_data', type_=dict)

    def setup(self):
        allocation_data = self.metadata['allocation_data']
        num_routes = allocation_data['num']
        num_existing_aircraft = allocation_data['num_existing']
        num_new_aircraft = allocation_data['num_new']
        num_aircraft = num_existing_aircraft + num_new_aircraft

        self.add_input('revenue', shape=(num_routes, ))
        self.add_input('pax_flt', shape=(num_routes, num_aircraft))
        self.add_input('tot_pax', shape=(num_routes, ))
        self.add_input('flt_day', shape=(num_routes, num_aircraft))

        # These are used in the final comparison spreadsheet.
        self.add_output('costs', shape=(num_routes, num_aircraft))

        # Fuelburn inputs
        for ind_nac in range(num_new_aircraft):
            for ind_rt in range(num_routes):
                fuelburn_name = self._get_fuelburn_name(ind_rt, ind_nac=ind_nac)
                self.add_input(fuelburn_name)

        # Blocktime inputs
        for ind_nac in range(num_new_aircraft):
            for ind_rt in range(num_routes):
                blocktime_name = self._get_blocktime_name(ind_rt, ind_nac=ind_nac)
                self.add_input(blocktime_name)

        self.add_output('profit', val=0.0)
        self.add_output('g_aircraft_new', shape=(num_new_aircraft, ))
        self.add_output('g_aircraft_exist', shape=(num_existing_aircraft, ))
        self.add_output('g_demand', shape=(num_routes, ))

        # TODO: Really would be better with anaytic derivs.
        #self.declare_partials(of='*', wrt='*')
        self.declare_partials(of='*', wrt=['revenue', 'pax_flt', 'tot_pax', '*fuel*', '*block*'])
        self.set_check_partial_options(wrt=['revenue'], step=1e-1)

    def _get_fuelburn_name(self, ind_rt, ind_ac=None, ind_nac=None):
        allocation_data = self.metadata['allocation_data']

        num_routes = allocation_data['num']
        num_existing_aircraft = allocation_data['num_existing']

        if ind_ac is not None:
            ind_nac = ind_ac - num_existing_aircraft

        index = ind_rt + ind_nac * num_routes
        return '{}_fuelburn_1e6_N'.format(index)

    def _get_blocktime_name(self, ind_rt, ind_ac=None, ind_nac=None):
        allocation_data = self.metadata['allocation_data']

        num_routes = allocation_data['num']
        num_existing_aircraft = allocation_data['num_existing']

        if ind_ac is not None:
            ind_nac = ind_ac - num_existing_aircraft

        index = ind_rt + ind_nac * num_routes
        return '{}_blocktime_hr'.format(index)

    def compute(self, inputs, outputs):
        allocation_data = self.metadata['allocation_data']
        num_routes = allocation_data['num']
        num_existing_aircraft = allocation_data['num_existing']
        num_new_aircraft = allocation_data['num_new']
        num_ac = num_existing_aircraft + num_new_aircraft

        trip = inputs['flt_day']
        rev = inputs['revenue']
        pax = inputs['pax_flt']
        tot_pax = inputs['tot_pax']
        cost_fuel_N = self.metadata['general_allocation_data']['cost_fuel_N']

        cost = 0.0
        g_aircraft_new = np.zeros((num_new_aircraft, ))
        g_aircraft_exist = np.zeros((num_existing_aircraft, ))

        #New aircraft
        for ind_nac in range(num_new_aircraft):
            name = allocation_data['new_names'][ind_nac]
            con_val = 0.0
            for jj in range(num_routes):
                x_kj = trip[jj, ind_nac]
                if x_kj > 0:
                    blocktime_name = self._get_blocktime_name(jj, ind_nac=ind_nac)
                    BH_kj = inputs[blocktime_name]*allocation_data['scale_fac']

                    MH_FH_kj = allocation_data['maint', name]
                    fuelburn_name = self._get_fuelburn_name(jj, ind_nac=ind_nac)
                    fuel_kj = inputs[fuelburn_name]*1e6
                    cost_kj = allocation_data['cost_other', name][jj]

                else:
                    cost_kj = 0.0
                    BH_kj=0.0
                    MH_FH_kj = 0.0
                    fuel_kj = 0.0

                #print('new', ind_nac, jj)
                #print('cost_kj', cost_kj)
                #print('BH_kj', BH_kj)
                #print('MH_FH_kj', MH_FH_kj)
                #print('fuel_kj', fuel_kj)
                #print('x_kj', x_kj)
                flt_cost = (cost_kj + fuel_kj*cost_fuel_N)*x_kj
                outputs['costs'][jj, ind_nac] = flt_cost
                cost += flt_cost
                con_val += x_kj*(BH_kj*(1.0 + MH_FH_kj) + 1.0)

            g_aircraft_new[ind_nac] = (con_val/(12.0*allocation_data['number', name]))

        #Existing aircraft
        for ind_ac in range(num_existing_aircraft):
            kk = num_new_aircraft + ind_ac
            name = allocation_data['existing_names'][ind_ac]
            con_val = 0.0
            for jj in range(num_routes):
                x_kj = trip[jj, kk]
                if x_kj > 0 and allocation_data['cost_other', name][jj] < 1.0e10:
                    #LF = int(round(10.0*pax[jj, kk]/allocation_data['capacity', name]))
                    ###TODO #This convention is different in matlab version (3D): dim1-routes,dim2-aircraft, dim3-LF
                    #cost_kj = allocation_data['TotCost_LF'][ind_ac, jj, LF]
                    #BH_kj = allocation_data['BH_LF'][ind_ac, jj, LF]
                    #MH_FH_kj = allocation_data['maint', name]
                    #fuel_kj = allocation_data['fuelburn_LF'][ind_ac, jj, LF]

                    # NOTE : Ignoring LF because of lack of data.
                    cost_kj = allocation_data['cost_other', name][jj]
                    BH_kj = allocation_data['block_time_hr', name][jj]
                    MH_FH_kj = allocation_data['maint', name]
                    fuel_kj = allocation_data['fuel_N', name][jj]

                else:
                    cost_kj = 0.0
                    BH_kj=0.0
                    MH_FH_kj = 0.0
                    fuel_kj = 0.0

                #print('exist', ind_ac, jj)
                #print('cost_kj', cost_kj)
                #print('BH_kj', BH_kj)
                #print('MH_FH_kj', MH_FH_kj)
                #print('fuel_kj', fuel_kj)

                flt_cost = (cost_kj + fuel_kj*cost_fuel_N)*x_kj
                outputs['costs'][jj, ind_nac + 1 + ind_ac] = flt_cost
                cost += flt_cost

                con_val += x_kj*(BH_kj*(1.0 + MH_FH_kj) + 1.0)

            g_aircraft_exist[ind_ac] = (con_val/(12.0*allocation_data['number', name]))

        #print('rev', rev)
        #print('cost', cost)
        outputs['profit'] = (np.sum(rev) - cost)/-1.0e3
        outputs['g_demand'] = tot_pax/allocation_data['demand']
        outputs['g_aircraft_new'] = g_aircraft_new
        outputs['g_aircraft_exist'] = g_aircraft_exist

    def compute_partials(self, inputs, partials):
        allocation_data = self.metadata['allocation_data']
        num_routes = allocation_data['num']
        num_existing_aircraft = allocation_data['num_existing']
        num_new_aircraft = allocation_data['num_new']
        num_ac = num_existing_aircraft + num_new_aircraft

        trip = inputs['flt_day']
        rev = inputs['revenue']
        pax = inputs['pax_flt']
        tot_pax = inputs['tot_pax']
        cost_fuel_N = self.metadata['general_allocation_data']['cost_fuel_N']

        pfact = 1.0/1.0e3

        #New aircraft
        for ind_nac in range(num_new_aircraft):
            name = allocation_data['new_names'][ind_nac]
            fact = 1.0/(12.0 * allocation_data['number', name])
            for jj in range(num_routes):
                x_kj = trip[jj, ind_nac]

                blocktime_name = self._get_blocktime_name(jj, ind_nac=ind_nac)
                BH_kj = inputs[blocktime_name]*allocation_data['scale_fac']

                MH_FH_kj = allocation_data['maint', name]
                fuelburn_name = self._get_fuelburn_name(jj, ind_nac=ind_nac)
                fuel_kj = inputs[fuelburn_name]*1e6
                cost_kj = allocation_data['cost_other', name][jj]

                # ii = jj * num_ac + ind_nac
                # partials['g_aircraft_new', 'flt_day'][ind_nac][ii] = (BH_kj*(1.0 + MH_FH_kj) + 1.0)*fact
                partials['g_aircraft_new', blocktime_name][ind_nac][0] = x_kj*(1.0 + MH_FH_kj)*fact

                # partials['profit', 'flt_day'][0][ii] = (cost_kj + fuel_kj*cost_fuel_N) * pfact
                partials['profit', fuelburn_name][0][0] = 1e6 * cost_fuel_N * x_kj* pfact

        #Existing aircraft
        for ind_ac in range(num_existing_aircraft):
            kk = num_new_aircraft + ind_ac
            name = allocation_data['existing_names'][ind_ac]
            fact = 1.0/(12.0 * allocation_data['number', name])
            for jj in range(num_routes):
                if allocation_data['cost_other', name][jj] < 1.0e10:

                    x_kj = trip[jj, kk]

                    # NOTE : Ignoring LF because of lack of data.
                    cost_kj = allocation_data['cost_other', name][jj]
                    BH_kj = allocation_data['block_time_hr', name][jj]
                    MH_FH_kj = allocation_data['maint', name]
                    fuel_kj = allocation_data['fuel_N', name][jj]

                    # partials['profit', 'flt_day'][0][jj] += (cost_kj + fuel_kj*cost_fuel_N) * pfact

                    # ii = jj * num_existing_aircraft + ind_ac
                    # partials['g_aircraft_exist', 'flt_day'][ind_ac][ii] = (BH_kj*(1.0 + MH_FH_kj) + 1.0)*fact



        partials['profit', 'revenue'][0][:] = -pfact
        partials['g_demand', 'tot_pax'] = np.diag(1.0/allocation_data['demand'])


if __name__ == '__main__':
    from openmdao.api import Problem, Group, IndepVarComp

    from prob_11_2_updated import allocation_data
    from prob_11_2_general_allocation import general_allocation_data

    prob = Problem()
    prob.model = model = Group()

    fuelburn_data = 1.0e15*np.array([[0.000000001355925,   6.688636363636363,   6.688636363636363],
                                     [0.000000000353455,   0.000000000185098,   0.000000000598590],
                                     [0.000000000500913,   6.688636363636363,   0.000000000851417],
                                     [0.000000000146787,   0.000000000075206,   0.000000000245963],
                                     [0.000000001086263,   6.688636363636363,   0.000000001805328],
                                     [0.000000000120387,   0.000000000061092,   0.000000000200078],
                                     [0.000000000345664,   0.000000000180974,   0.000000000585469],
                                     [0.000000000369548,   0.000000000193490,   0.000000000625277],
                                     [0.000000000232424,   0.000000000121118,   0.000000000394406],
                                     [0.000000000699532,   6.688636363636363,   0.000000001178648],
                                     [0.000000000577013,   6.688636363636363,   0.000000000978270]])

    BH_data = np.array([14.74320739, 4.56254104, 6.31734857, 1.86057744, 12.84726843, 1.5558875, 4.46579297, 4.75964448, 3.01946347 ,8.52760936, 7.18798118])

    dd = model.add_subsystem('dummy', IndepVarComp(), promotes=['*'])
    for j in range(11):
        dd.add_output('{}_blocktime_hr'.format(j), BH_data[j])
        dd.add_output('{}_fuelburn_1e6_N'.format(j), fuelburn_data[j, 0])

    model.add_subsystem('p_flt', IndepVarComp('flt_day', val=allocation_data['flt_day'].T ),
                        promotes=['*'])

    model.add_subsystem('revenue', RevenueManager(general_allocation_data=general_allocation_data,
                                                  allocation_data=allocation_data),
                        promotes=['*'])
    model.add_subsystem('profit', Profit(general_allocation_data=general_allocation_data,
                                         allocation_data=allocation_data),
                        promotes=['*'])

    xC0_rev = 1.0e3*np.array([[ 4.3460,    1.3430,    1.7560,    0.7062,    3.6570,    0.6189,    1.3200,    1.3890,    0.9810,    2.4250,    1.9650],
                              [ 2.6318,    0.4475,    0.5851,    0.2357,    1.2197,    0.2159,    0.4400,    0.4629,    0.3269,    0.7980,    0.6416],
                              [ 6.0180,    1.1577,    1.5140,    0.6311,    3.1820,    0.7101,    1.1561,    1.2043,    0.8479,    2.1036,    1.6272],
                              [ 3.7277,    0.6471,    0.8467,    0.3331,    1.7368,    0.3450,    0.6442,    0.6730,    0.4673,    1.1667,    0.9138],
                              [ 0.3000,    3.1920,    4.1120,    2.2420,    1.2480,    0.3000,    0.7160,    1.8960,    2.2640,    0.4160,    0.4160]])

    dd.add_output('revenue:x1', xC0_rev[0, :])
    dd.add_output('revenue:y1', xC0_rev[1, :])
    dd.add_output('revenue:x2', xC0_rev[2, :])
    dd.add_output('revenue:y2', xC0_rev[3, :])
    dd.add_output('revenue:z1', xC0_rev[4, :])

    #prob['revenue:x1'] = xC0_rev[0, :]
    #prob['revenue:y1'] = xC0_rev[1, :]
    #prob['revenue:x2'] = xC0_rev[2, :]
    #prob['revenue:y2'] = xC0_rev[3, :]
    #prob['revenue:z1'] = xC0_rev[4, :]

    model.add_design_var('revenue:x1')
    model.add_design_var('revenue:y1')
    model.add_design_var('revenue:x2')
    model.add_design_var('revenue:y2')
    model.add_design_var('revenue:z1')
    model.add_objective('profit')
    model.add_constraint('g_demand', upper=0.0)
    model.add_constraint('g_aircraft_new', upper=0.0)
    model.add_constraint('g_aircraft_exist', upper=0.0)
    prob.setup()

    prob.run()
    print('Revenue')
    print('---------')
    print('pax_flt', prob['pax_flt'])
    print('revenue', prob['revenue'])
    print('tot_pax', prob['tot_pax'])

    print('Profit/Con')
    print('------------')
    print('profit', prob['profit'])
    print('g_demand', prob['g_demand'])
    print('g_aircraft_new', prob['g_aircraft_new'])
    print('g_aircraft_exist', prob['g_aircraft_exist'])

    prob.check_partials(step=1.0e-5, compact_print=True)
    #prob.check_partials(comps=['revenue'], compact_print=True)

    prob.check_totals(compact_print=True, step=1e-1)
