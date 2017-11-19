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


class RevenueManager(ExplicitComponent):
    """
    Uses revenue model to output passengers demand per route.
    """
    def initialize(self):
        self.metadata.declare('general_allocation_data', type_=dict)
        self.metadata.declare('allocation_data', type_=dict)

    def setup(self):
        num_routes = self.metadata['allocation_data']['num']
        num_existing_aircraft = self.metadata['allocation_data']['num_existing']
        num_new_aircraft = self.metadata['allocation_data']['num_new']
        num_aircraft = num_existing_aircraft + num_new_aircraft

        self.add_input('revenue:x1', shape=(num_routes, ))
        self.add_input('revenue:y1', shape=(num_routes, ))
        self.add_input('revenue:x2', shape=(num_routes, ))
        self.add_input('revenue:y2', shape=(num_routes, ))
        self.add_input('revenue:z1', shape=(num_routes, ))

        self.add_input('flt_day', shape=(num_routes, num_aircraft))

        self.add_output('pax_flt', shape=(num_routes, num_aircraft))

        # TODO: Really would be better with anaytic derivs.
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        num_routes = self.metadata['allocation_data']['num']
        num_existing_aircraft = self.metadata['allocation_data']['num_existing']
        num_new_aircraft = self.metadata['allocation_data']['num_new']
        num_ac = num_existing_aircraft + num_new_aircraft

        seats = []
        for key in allocation_data['existing_names'] + allocation_data['new_names']:
            seats.append(self.metadata['allocation_data']['capacity', key])
        seats = np.array(seats)

        trip = inputs['flt_day']
        x1 = inputs['revenue:x1']
        y1 = inputs['revenue:y1']
        x2 = inputs['revenue:x2']
        y2 = inputs['revenue:y2']
        z1 = inputs['revenue:z1']

        # Total number of seats available per day for a route.
        max_avail_seats = seats.dot(trip.T)

        # Main Revenue calculation.
        for jj in range(num_routes):
            if max_avail_seats[jj] > 0:
                rev_j, totnacc_j, nacc_j, p_j, ntot_j = calc_revenue(x1[jj], y1[jj], x2[jj],
                                                                     y2[jj], z1[jj], inits.alpha[jj,:],
                                                                     inits.beta[jj,:], max_avail_seats[jj])
                rev[jj] = rev_j
                nacc[jj,:] = nacc_j
                p[jj,:] = p_j
                ntot[jj,:] = ntot_j
                for kk in range(num_ac):
                    x_kj = trip[kk*num_route + jj]
                    if x_kj>0:
                        pax[kk,jj] = min(seat[kk], seat[kk]*totnacc_j/max_avail_seats[jj])
            else:
                totnacc_j = 0.0

            tot_pax[jj] = totnacc_j

            if pax[0,jj]>0:
                mission_route = np.append(mission_route,inits.network.route[jj])
                mission_pax = np.append(mission_pax,pax[0,jj])

        print('done')

'''
def obj_cons_calc(xC, xI, inits):
    """ Calculate fleet-level profit of the airline and the problem constraints"""
    # xC = xClb + xC_val.flatten()*(xCub - xClb)
    newac = inits.newac
    existac = inits.existac
    filename = newac.filename
    num_route = inits.network.num_route
    price = inits.network.price
    dem = inits.network.dem
    num_ac = len(inits.existAC_index)+1
    trip = xI
    seat = inits.seat #np.concatenate((newac.seat_cap,existac.seat_cap), axis=0)
    AC_num = np.concatenate((newac.AC_num_new,existac.AC_num), axis=0)
    num_xC_des = inits.num_xC_des
    FP = inits.FP
    FPperlb = FP/6.84
    scale_fac = inits.scale_fac

    max_avail_seats = np.zeros((num_route,))
    for jj in range(num_route):
        av_seats = 0
        for kk in range(num_ac):
            av_seats += seat[kk]*trip[kk*num_route + jj]
        max_avail_seats[jj] = av_seats
    # Continuous data extract
    x_con = xC[:num_xC_des]
    # Revenue management variables
    RMS_var = xC[num_xC_des:]
    x1 = RMS_var[:num_route]
    y1 = RMS_var[num_route:2*num_route]
    x2 = RMS_var[num_route*2:3*num_route]
    y2 = RMS_var[num_route*3:4*num_route]
    z1 = RMS_var[num_route*4:]

    rev = np.zeros((num_route,))
    nacc = np.zeros((num_route,2))
    p = np.zeros((num_route,2))
    ntot = np.zeros((num_route,2))
    pax = np.zeros((num_ac,num_route))
    tot_pax = np.zeros((num_route,1))
    mission_route = newac.DESRNG
    mission_pax = newac.seat_cap
    for jj in range(num_route):
        if max_avail_seats[jj]>0:
            rev_j, totnacc_j, nacc_j, p_j, ntot_j = RevenueManagementSys_Det(x1[jj],y1[jj],x2[jj],y2[jj],z1[jj],inits.alpha[jj,:],inits.beta[jj,:],max_avail_seats[jj])
            rev[jj] = rev_j
            nacc[jj,:] = nacc_j
            p[jj,:] = p_j
            ntot[jj,:] = ntot_j
            for kk in range(num_ac):
                x_kj = trip[kk*num_route + jj]
                if x_kj>0:
                    pax[kk,jj] = min(seat[kk], seat[kk]*totnacc_j/max_avail_seats[jj])
        else:
            totnacc_j = 0.0

        tot_pax[jj] = totnacc_j

        if pax[0,jj]>0:
            mission_route = np.append(mission_route,inits.network.route[jj])
            mission_pax = np.append(mission_pax,pax[0,jj])


#################################################################################
# TODO: This part of the code needds to be updated for AMD. Just provide the mission performance data of the new aircraft
    # Generate input file for FLOPS
    FLOPSInputGen(x_con, mission_route, mission_pax, newac, FP, filename)

    # Execute FLOPS
    try:
        cmmndline = "./flops <"+filename+".in > "+filename+".out"
        out = subprocess.check_output(['bash','-c',cmmndline])
    except Exception as err:
        #raise AnalysisError("FLOPS failed to converge")
        print("FLOPS failed to converge!")
        TD = 10000.0
        LD = 10000.0
        TOC = 1e5*np.ones((num_route+1,1))
        BH = 25.0*np.ones((num_route+1,1))

    # Parse output file from FLOPS
    TD, LD, TOC, BH, nanC = ReadFLOPSOutput(filename,mission_route)

    cost_1j = TOC[1:]
    BH_1j = BH[1:]
    fuelburn_1j = fuelburn[1:]
    # TODO:SR-Need these data from AMD
#################################################################################


'''

if __name__ == '__main__':
    from openmdao.api import Problem, Group

    from prob_11_2_updated import allocation_data
    from amd_om.allocation.airline_networks.general_allocation_data import general_allocation_data

    prob = Problem()
    prob.model = model = Group()

    model.add_subsystem('Revenue', RevenueManager(general_allocation_data=general_allocation_data,
                                                  allocation_data=allocation_data),
                        promotes=['*'])

    prob.setup()

    prob.run()
