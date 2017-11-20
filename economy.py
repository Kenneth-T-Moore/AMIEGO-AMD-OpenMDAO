"""
Revenue Management System for AMD optimization.
"""
from __future__ import print_function

from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent


def calc_revenue(x1, y1, x2, y2, z1, alpha, beta, Cap):
    # Attractiveness coefficients for fare classes
    a = np.array([0.864, -0.038])
    b = np.array([-0.02, 0.016])
    c = np.array([0.009, 0.008])

    x = np.array([x1, x2])
    y = np.array([y1, y2])

    mu = alpha - beta*y
    p = 1.0/(1.0 + np.exp(a - b*y + c*x))

    ntot = mu
    nacc = np.zeros((2,))
    R = np.zeros((2,))

    nacc[0] = min(ntot[0],z1)
    R[0] = (p[0]*x[0] + (1.0 - p[0])*y[0])*nacc[0]

    nacc[1] = min(ntot[1],(Cap - nacc[0]))
    R[1] = (p[1]*x[1] + (1.0 - p[1])*y[1])*nacc[1]

    Rev = np.sum(R)
    sum_nacc = np.sum(nacc)

    return Rev, sum_nacc, nacc, p, ntot


def demand_bounds(dem, price, range_route):
    LF1 = 0.6; LF2 = 1.0-LF1
    k=1.2

    p2 = price*k
    p1 = price*(1.0 - k*LF2)/LF1

    alp1, bet1 = price_elasticity(LF1,dem,p1,range_route)
    alp2, bet2 = price_elasticity(LF2,dem,p2,range_route)

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
        alloc_data = self.metadata['allocation_data']
        num_routes = alloc_data['num']
        num_existing_aircraft = alloc_data['num_existing']
        num_new_aircraft = alloc_data['num_new']
        num_aircraft = num_existing_aircraft + num_new_aircraft

        self.add_input('revenue:x1', shape=(num_routes, ))
        self.add_input('revenue:y1', shape=(num_routes, ))
        self.add_input('revenue:x2', shape=(num_routes, ))
        self.add_input('revenue:y2', shape=(num_routes, ))
        self.add_input('revenue:z1', shape=(num_routes, ))

        self.add_input('flt_day', shape=(num_aircraft, num_routes))

        self.add_output('pax_flt', shape=(num_routes, num_aircraft))
        self.add_output('revenue', shape=(num_routes, ))
        self.add_output('tot_pax', shape=(num_routes, ))

        # TODO: Really would be better with anaytic derivs.
        self.declare_partials(of='*', wrt='*', method='fd')

        # Calculate Demand bounds for each route.
        self.alpha = np.empty((num_routes, 2))
        self.beta = np.empty((num_routes, 2))
        demand = alloc_data['demand']
        price = alloc_data['price_pax', 'CRM']
        route = alloc_data['range_km']
        for jj in range(num_routes):
            alp, bet = demand_bounds(demand[jj], price[jj], route[jj])
            self.alpha[jj, :] = alp
            self.beta[jj, :] = bet

    def compute(self, inputs, outputs):
        alloc_data = self.metadata['allocation_data']
        num_routes = alloc_data['num']
        num_existing_aircraft = alloc_data['num_existing']
        num_new_aircraft = alloc_data['num_new']
        num_ac = num_existing_aircraft + num_new_aircraft

        seats = []
        for key in allocation_data['names']:
            seats.append(alloc_data['capacity', key])
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
        max_avail_seats = seats.dot(trip)

        # Main Revenue calculation.
        rev = np.zeros((num_routes, ))
        pax = np.zeros((num_routes, num_ac))
        tot_pax = np.empty((num_routes, ))
        for jj in range(num_routes):
            if max_avail_seats[jj] > 0:
                rev_j, totnacc_j, _, _, _ = calc_revenue(x1[jj], y1[jj], x2[jj], y2[jj], z1[jj],
                                                         alpha[jj, :], beta[jj, :],
                                                         max_avail_seats[jj])
                rev[jj] = rev_j
                for kk in range(num_ac):
                    x_kj = trip[kk, jj]
                    if x_kj > 0:
                        pax[jj, kk] = min(seats[kk], seats[kk]*totnacc_j/max_avail_seats[jj])
            else:
                totnacc_j = 0.0

            tot_pax[jj] = totnacc_j

        outputs['pax_flt'] = pax
        outputs['revenue'] = rev
        outputs['tot_pax'] = tot_pax


class Profit(ExplicitComponent):
    """
    Calculate values for airline profit and constraints.
    """
    def initialize(self):
        self.metadata.declare('general_allocation_data', type_=dict)
        self.metadata.declare('allocation_data', type_=dict)

    def setup(self):
        alloc_data = self.metadata['allocation_data']
        num_routes = alloc_data['num']
        num_existing_aircraft = alloc_data['num_existing']
        num_new_aircraft = alloc_data['num_new']
        num_aircraft = num_existing_aircraft + num_new_aircraft

        self.add_input('revenue', shape=(num_routes, ))
        self.add_input('pax_flt', shape=(num_routes, num_aircraft))
        self.add_input('tot_pax', shape=(num_routes, ))
        self.add_input('flt_day', shape=(num_aircraft, num_routes))

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
        self.declare_partials(of='*', wrt='*', method='fd')

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
                x_kj = trip[ind_nac, jj]
                if x_kj > 0:
                    blocktime_name = self._get_blocktime_name(jj, ind_nac=ind_nac)
                    BH_kj = inputs[blocktime_name]
                    #BH_kj = BH_1j[dd]*inits.scale_fac

                    MH_FH_kj = allocation_data['maint', name]
                    fuelburn_name = self._get_fuelburn_name(jj, ind_nac=ind_nac)
                    fuel_kj = inputs[fuelburn_name]
                    cost_kj = allocation_data['cost_other', name][jj]
                    print('cost', cost_kj)

                else:
                    cost_kj = 0.0
                    BH_kj=0.0
                    MH_FH_kj = 0.0

                cost += (cost_kj + fuel_kj*cost_fuel_N)*x_kj
                con_val += x_kj*(BH_kj*(1.0 + MH_FH_kj) + 1.0)

            g_aircraft_new[ind_nac] = (con_val/(12.0*allocation_data['number', name]))

        #Existing aircraft
        for ind_ac in range(num_existing_aircraft):
            kk = num_new_aircraft + ind_ac
            name = allocation_data['existing_names'][ind_ac]
            con_val = 0.0
            for jj in range(num_routes):
                x_kj = trip[kk, jj]
                if x_kj > 0:
                    LF = int(round(10.0*pax[jj, kk]/allocation_data['capacity', name]))
                    ##TODO #This convention is different in matlab version (3D): dim1-routes,dim2-aircraft, dim3-LF
                    cost_kj = allocation_data['TotCost_LF'][ind_ac, jj, LF-1]
                    BH_kj = allocation_data['BH_LF'][ind_ac, jj, LF-1]
                    MH_FH_kj = allocation_data['maint', name]
                    fuel_kj = allocation_data['fuelburn_LF'][ind_ac, jj, LF-1]
                    print('cost', cost_kj)
                else:
                    cost_kj = 0.0
                    BH_kj=0.0
                    MH_FH_kj = 0.0

                cost += (cost_kj + fuel_kj*cost_fuel_N)*x_kj
                con_val += x_kj*(BH_kj*(1.0 + MH_FH_kj) + 1.0)

            g_aircraft_exist[ind_ac] = (con_val/(12.0*allocation_data['number', name]))

        outputs['profit'] = (np.sum(rev) - cost)/-1.0e3
        outputs['g_demand'] = tot_pax/allocation_data['demand']
        outputs['g_aircraft_new'] = g_aircraft_new
        outputs['g_aircraft_exist'] = g_aircraft_exist


if __name__ == '__main__':
    from openmdao.api import Problem, Group, IndepVarComp

    from prob_11_2_updated import allocation_data
    from prob_11_2_general_allocation import general_allocation_data

    prob = Problem()
    prob.model = model = Group()

    dd = model.add_subsystem('dummy', IndepVarComp(), promotes=['*'])
    for j in range(11):
        dd.add_output('{}_blocktime_hr'.format(j), 1.0)
        dd.add_output('{}_fuelburn_1e6_N'.format(j), 1.0)

    model.add_subsystem('p_flt', IndepVarComp('flt_day', val=allocation_data['flt_day'] ))

    model.add_subsystem('revenue', RevenueManager(general_allocation_data=general_allocation_data,
                                                  allocation_data=allocation_data),
                        promotes=['*'])
    model.add_subsystem('profit', Profit(general_allocation_data=general_allocation_data,
                                         allocation_data=allocation_data),
                        promotes=['*'])

    prob.setup()

    xC0_rev = 1.0e3*np.array([[4.345998241,	1.342998121,	1.75599824,	0.70620896,	3.656998241,	0.618935819,	1.319998154,	1.388998192,	0.980982907,	2.424998241,	1.964998241,],
                              [2.631796491,	0.447472483,	0.585112984,	0.23568446,	1.219687876,	0.215893558,	0.440036612,	0.462931229,	0.326884785,	0.798004172,	0.641578808,],
                              [6.018,	1.157666785,	1.513995811,	0.631128339,	3.182006273,	0.710131995,	1.156074986,	1.204342339,	0.847851248,	2.103614992,	1.627245041,],
                              [3.727684522,	0.647146408,	0.846747348,	0.333136529,	1.736822468,	0.344980839,	0.644194989,	0.67295188,	0.467312679,	1.16673843,	0.9138423,],
                              [0.299998241,	3.213998241,	4.111998241,	2.497998241,	1.247998241,	0.162,	0.715998241,	1.847998241,	2.173998241,	0.415998241,	0.415998241]])

    prob['revenue:x1'] = xC0_rev[0, :]
    prob['revenue:y1'] = xC0_rev[1, :]
    prob['revenue:x2'] = xC0_rev[2, :]
    prob['revenue:y2'] = xC0_rev[3, :]
    prob['revenue:z1'] = xC0_rev[4, :]

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
