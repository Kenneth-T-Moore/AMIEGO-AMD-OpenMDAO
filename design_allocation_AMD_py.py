"""Aircraft Design and Airline Allocation using AMD to generate aircraft performance data.

- X_desgin,mission + 5*num_route - continuous variables

- num_route*num_ac - inetger type variables

"""
from __future__ import print_function

import os
import subprocess
import sys

from mpi4py import MPI

import numpy as np

from openmdao.components.external_code import ExternalCode
from openmdao.core.analysis_error import AnalysisError

if sys.platform == 'win32':
    carriage = '\r\n'
else:
    carriage = '\n'

def FLOPSInputGen(x_con, mission_route, mission_pax, newac, FP, filename):
    '''Generate the input deck for FLOPS'''
    wt_pax = 165.0
    NPF = np.round(0.07*newac.seat_cap)
    if np.mod(NPF, 2) == 1:
        NPF = NPF-1
    NPT = newac.seat_cap - NPF

    # Write in a new file
    titleline = 'Input deck ' + filename
    fname = filename + '.in'
    fid = open(fname,'w')
    fid.write(titleline+' ' + carriage)
    fid.write(' $OPTION ' + carriage)
    fid.write('  IOPT=1, IANAL=3, ICOST=1, ' + carriage)
    fid.write(' $END ' + carriage)
    fid.write(' $WTIN ' + carriage)
    fid.write('  DGW='+str(newac.GW)+', ' + carriage)
    fid.write('  VMMO='+str(0.82)+', ' + carriage)
    fid.write('  DIH='+str(6.0)+', ' + carriage)
    fid.write('  HYDPR='+str(3000.0)+', ' + carriage)
    fid.write('  WPAINT='+str(0.03)+', ' + carriage)
    fid.write('  XL='+str(129.5)+', ' + carriage)
    fid.write('  WF='+str(12.33)+', ' + carriage)
    fid.write('  DF='+str(13.5)+', ' + carriage)
    fid.write('  XLP='+str(98.5)+', ' + carriage)
    fid.write('  SHT='+str(353.1)+', ' + carriage)
    fid.write('  SWPHT='+str(284.2)+', ' + carriage)
    fid.write('  ARHT='+str(353.1)+', ' + carriage)
    fid.write('  TRHT='+str(0.28)+', ' + carriage)
    fid.write('  TCHT='+str(0.09)+', ' + carriage)
    fid.write('  SVT='+str(284.2)+', ' + carriage)
    fid.write('  SWPVT='+str(39.4)+', ' + carriage)
    fid.write('  ARVT='+str(1.24)+', ' + carriage)
    fid.write('  TRVT='+str(0.39)+', ' + carriage)
    fid.write('  TCVT='+str(0.09)+', ' + carriage)
    fid.write('  NEW='+str(int(newac.NEW))+', ' + carriage)
    fid.write('  NPF='+str(int(NPF))+', ' + carriage)
    fid.write('  NPT='+str(int(NPT))+', ' + carriage)
    # fid.write('  WPPASS='+str(wt_pax)+', ' + carriage)
    # fid.write('  BPP='+str(bag_pax)+', ' + carriage)
    # fid.write('  CARGOF='+str(5500)+', ' + carriage)
    fid.write('  WSRV=1.8, ' + carriage)
    fid.write('  IFUFU=1, ' + carriage)
    fid.write('  MLDWT=0,  WAPU=1.0,  WHYD=1.0,  ' + carriage)
    fid.write(' $END ' + carriage)

    fid.write(' $CONFIN ' + carriage)
    fid.write('  DESRNG='+ ','.join((str(val) for val in newac.DESRNG)) +', ' + carriage)
    fid.write('  GW='+str(newac.GW)+', ' + carriage)
    fid.write('  AR='+str(x_con[0])+', ' + carriage)
    fid.write('  TR='+str(x_con[1])+', ' + carriage)
    fid.write('  TCA='+str(x_con[2])+', ' + carriage)
    fid.write('  SW='+str(x_con[3])+', ' + carriage)
    fid.write('  SWEEP='+str(x_con[4])+', ' + carriage)
    fid.write('  THRUST='+str(x_con[5])+', ' + carriage)
    fid.write('  VCMN='+str(0.787)+', ' + carriage)
    fid.write('  CH='+str(41000.0)+', ' + carriage)
    fid.write('  HTVC='+str(2.84)+', ' + carriage)
    fid.write('  VTVC='+str(0.24)+', ' + carriage)
    fid.write('  OFG=1., OFF=0., OFC=0., ' + carriage)
    fid.write(' $END ' + carriage)

    fid.write(' $AERIN ' + carriage)
    fid.write('  VAPPR='+str(142.0)+', ' + carriage)
    fid.write('  AITEK='+str(1.819)+', ' + carriage)
    fid.write('  E=0.93365, ' + carriage)
    fid.write(' $END ' + carriage)

    fid.write(' $COSTIN ' + carriage)
    fid.write('  ROI='+str(7.0)+', ' + carriage)
    fid.write('  FARE='+str(0.10)+', ' + carriage)
    fid.write('  FUELPR='+str(FP)+', ' + carriage)
    fid.write('  LF='+str(0.8)+', ' + carriage)
    fid.write('  DEVST='+str(2012)+', ' + carriage)
    fid.write('  DYEAR='+str(2017)+', ' + carriage)
    fid.write('  PLMQT='+str(2016)+', ' + carriage)
    fid.write(' $END ' + carriage)

    fid.write(' $ENGDIN  ' + carriage)
    fid.write('  IDLE=1, IGENEN=1, ' + carriage)
    fid.write('  MAXCR=1, NGPRT=0, ' + carriage)
    fid.write(' $END  ' + carriage)

    fid.write(' $ENGINE ' + carriage)
    fid.write('  IENG=2, IPRINT=0, ' + carriage)
    fid.write('  OPRDES='+str(29.5)+', ' + carriage)
    fid.write('  FPRDES='+str(1.67)+', ' + carriage)
    fid.write('  TETDES='+str(2660.0)+', ' + carriage)
    fid.write(' $END ' + carriage)

    fid.write(' $MISSIN ' + carriage)
    fid.write('  IFLAG=2, ' + carriage)
    fid.write('  IRW=1, ' + carriage)
    fid.write('  ITTFF=1, ' + carriage)
    fid.write('  TAKOTM=0.4, ' + carriage)
    fid.write('  TAXOTM=10, ' + carriage)
    fid.write('  TAXITM=10, ' + carriage)
    fid.write('  FWF=-1, ' + carriage)
    fid.write('  THOLD=0.05, ' + carriage)
    fid.write('  RESRFU=0.05, ' + carriage)
    fid.write(' $END ' + carriage)

    fid.write('START ' + carriage)
    fid.write('CLIMB ' + carriage)
    fid.write('CRUISE ' + carriage)
    fid.write('DESCENT ' + carriage)
    fid.write('END ' + carriage)

    for jj in range(1, len(mission_route)):
        fid.write('$RERUN ' + carriage)
        fid.write('  mywts = 1, wsr = 0., twr = 0. , ' + carriage)
        fid.write('  desrng='+str(mission_route[jj])+', ' + carriage)
        if mission_route[jj]<=900.0:
            bag_pax = 35.0
        elif mission_route[jj]>900.0 and mission_route[jj]<=2900.0:
            bag_pax = 40.0
        else:
            bag_pax = 44.0
        payload = mission_pax[jj]*(wt_pax + bag_pax)
        fid.write('  paylod='+str(payload)+', ' + carriage)
        fid.write('$END ' + carriage);

        fid.write(' $MISSIN ' + carriage)
        fid.write('  IFLAG=0, ' + carriage)
        fid.write(' $END ' + carriage)

        fid.write('START ' + carriage)
        fid.write('CLIMB ' + carriage)
        fid.write('CRUISE ' + carriage)
        fid.write('DESCENT ' + carriage)
        fid.write('END ' + carriage)
    fid.close


def ReadFLOPSOutput(filename, ranges):
    """Read from the FLOPS output file.

    Args
    ----
    filename : string
        Filename stem for the FLOPS output file.
    ranges : ndarray
        mission ranges in nmi (incl. design mission).

    Returns
    -------
    TD : float
        Take off distance of the design mission.
    LD : float
        Landing distance of the design mission.
    TOC : ndarray
        Total operating cost for each mission (incl. design mission).
    BH : ndarray
        Block hours for each mission (incl. design mission).
    nanC : digit
        Number of failed missions
    """
    TD = 1.0e4; LD = 1.0e4
    TOC = 1.0e5*np.ones((len(ranges),1));
    BH = 25.0*np.ones((len(ranges),1))

    fname = filename + '.out'
    fid = open(fname,'r')
    ind = 0; delta = 1.0; nanC = 0

    line = fid.readline()
    while line != '':
        if len(line)>22 and line[:23] == ' DIRECT OPERATING COSTS':
            ind += 1
            line = fid.readline()
            words = line.split()
            try:
                DOC = float(words[-1])
                if np.isnan(DOC):
                    DOC = 0.5e5
            except:
                DOC = 0.5e5

            # Proceed to line for DOCperBH
            line = fid.readline()
            words = line.split()
            try:
                DOCperBH = float(words[-1])
                if np.isnan(DOCperBH):
                    DOCperBH = 0.5e5
            except:
                DOCperBH = 1000.0

            # Proceed to line for IOC
            for cc in range(24):
                line = fid.readline()
            words = line.split()
            try:
                IOC = float(words[-1])
                if np.isnan(IOC):
                    IOC = 0.5e5
            except:
                IOC = 0.5e5

            # Proceed to line for mission fuelburn
            for cc in range(36):
                line = fid.readline()
            words = line.split()

            try:
                fb = float(words[0])
                if np.isnan(fb):
                    fb = 1.0e5
            except:
                fb = 1.0e5

            try:
                RNG = float(words[1])
                if np.isnan(RNG):
                    RNG = 1.0e5
            except:
                RNG = 1.0e5

            try:
                td = float(words[3])
                if np.isnan(td):
                    td = 1.0e4
            except:
                td = 1.0e4

            try:
                ld = float(words[4])
                if np.isnan(ld):
                    ld = 1.0e4
            except:
                ld = 1.0e4

            while (ind <= len(ranges) and \
            (RNG<(ranges[ind-1]-delta) or RNG>(ranges[ind-1]+delta))):
                ind += 1; nanC += 1

            if ind <= len(ranges):
                TOC[ind-1] = DOC+IOC
                BH[ind-1] = DOC/DOCperBH
                if ind == 1:
                    # Return design mission fuelburn, take-off & landing field lengths
                    FB = fb; TD = td; LD = ld

        line = fid.readline()
    fid.close
    #os.remove(fname)
    # fnameIN = filename + '.in'
    #os.remove(fnameIN)
    return TD, LD, TOC, BH, nanC


class Initialize():
    def __init__(self, num_route, existAC_index):
        #TODO: Read this directly from below
        if num_route == 3:
            self.route_index = route_index = np.array([7, 6, 4])
        elif num_route == 11:
            self.route_index = route_index = np.array(range(11))+1
        self.num_route = num_route
        self.num_ac = num_ac = len(existAC_index) + 1
        self.existAC_index = existAC_index
        self.newac = NewAC(route_index)
        self.existac = ExistingAC(existAC_index, route_index)
        self.network = NetworkData(num_route, route_index)
        self.seat = np.concatenate((self.newac.seat_cap, self.existac.seat_cap), axis=0)
        self.num_xC_des = 6 #TODO: SR-Change this!
        self.FP = 1.41 #[$/lb] Fuel price

        self.num_xI = num_route*num_ac
        self.xIlb= np.zeros((self.num_xI, ))
        self.xIub= np.zeros((self.num_xI, ))

        self.scale_fac = 1.0

        # Initialize the continuous and integer variables and define their bounds
        self.alpha = np.zeros((num_route,2))
        self.beta = np.zeros((num_route,2))
        for jj in range(num_route):
            [alp,bet] = demand_bounds(self.network.dem[jj],self.network.price[jj],self.network.route[jj])
            self.alpha[jj,:] = alp
            self.beta[jj,:] = bet

        xC0_rev = 1.0e3*np.array([4.345998241,	1.342998121,	1.75599824,	0.70620896,	3.656998241,	0.618935819,	1.319998154,	1.388998192,	0.980982907,	2.424998241,	1.964998241,\
        	                       2.631796491,	0.447472483,	0.585112984,	0.23568446,	1.219687876,	0.215893558,	0.440036612,	0.462931229,	0.326884785,	0.798004172,	0.641578808,\
                                   	6.018,	1.157666785,	1.513995811,	0.631128339,	3.182006273,	0.710131995,	1.156074986,	1.204342339,	0.847851248,	2.103614992,	1.627245041,\
                                    	3.727684522,	0.647146408,	0.846747348,	0.333136529,	1.736822468,	0.344980839,	0.644194989,	0.67295188,	0.467312679,	1.16673843,	0.9138423,\
                                        	0.299998241,	3.213998241,	4.111998241,	2.497998241,	1.247998241,	0.162,	0.715998241,	1.847998241,	2.173998241,	0.415998241,	0.415998241])

        self.num_xC_rev = len(xC0_rev)
        #Initial starting point set to baseline solution
        self.xC0 = np.concatenate((np.array([9.4,0.159,0.1338,1345.5,25.0,24200.0]) , xC0_rev),axis=0) #TODO: SR-change this
        self.num_xC = len(self.xC0)

        for kk in range(num_ac):
            for jj in range(num_route):
                self.xIub[kk*num_route + jj] = np.ceil(self.network.dem[jj]/(0.8*self.seat[kk]))


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


def price_elasticity(LF,avail_seat,price,ran):
    alt_mode = 1 #Set this to 1 if alternate mode of transportation is available, otherwise 0

    dem = LF*avail_seat
    RVector_km = ran*1.852
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


class NewAC():
    def __init__(self, route_index):
        # Intermediate inputs about the 'yet-to-be-designed' aircraft
        # self.num_des = 6 #Number of aircraft design variables
        self.filename = 'AC_New' #A better version of B737-8ish aircraft
        if len(route_index) == 3:
            self.AC_num_new = np.array([3])
        elif len(route_index) == 11:
            self.AC_num_new = np.array([10])

        self.MH_new = np.array([0.866])
        self.DESRNG = np.array([8000.0])
        self.GW = 516000.0
        self.NEW = 2
        self.seat_cap = np.array([300.0])

        #Estimates of costs & BH for failed cases, Required for preopt screening
        #TODO: SR-Assumes independednt of load factor -future release will include variation due to load factor
        cost_new_data = 1.0e4*np.array([[130981.01],
                                        [42877.13574],\
                                        [53463.95286],\
                                        [24990.15611],\
                                        [112051.5746],\
                                        [22593.39838],\
                                        [42247.3911],\
                                        [44153.44306],\
                                        [32840.54423],\
                                        [74353.72696],\
                                        [59001.04963]])

        cost_new_data = np.repeat(cost_new_data,11,axis=1)

        BH_new_data = np.array([[14.74320739],\
                                [4.56254104],\
                                [6.31734857],\
                                [1.86057744],\
                                [12.84726843],\
                                [1.5558875],\
                                [4.46579297],\
                                [4.75964448],\
                                [3.01946347],\
                                [8.52760936],\
                                [7.18798118]])

        BH_new_data = np.repeat(BH_new_data,11,axis=1)

        fuel_new_data = np.array([[304081.1205],\
        	                      [79266.07176],\
                                  [112335.1755],\
                                  [32918.51254],\
                                  [243606.3005],\
                                  [26998.13761],\
                                  [77519.0318],\
                                  [82875.30071],\
                                  [52123.65138],\
                                  [156877.674],\
                                  [129401.5309]])

        fuel_new_data = np.repeat(fuel_new_data,11,axis=1)

        self.cost_new = cost_new_data[route_index-1,:]
        self.BH_new_data = BH_new_data[route_index-1,:]
        self.fuel_new_data = fuel_new_data[route_index-1,:]
        self.TD_new = 8171.0
        self.LD_new = 6506.0


class ExistingAC():
    def __init__(self,existAC_index, route_index):
        AC_name = ['B737-800','B747-800']
        if len(route_index) == 3:
            AC_num=np.array([7, 7])
        elif len(route_index) == 11:
            AC_num=np.array([18, 25])

        seat_cap=np.array([162.0,416.0])
        des_range=np.array([2940.0,7262])
        MH_FH=np.array([0.866, 0.866])
        #Data in the format Dim1: Aircraft type, Dim2: Route, Dim3: Load factor
        #TODO: This cost is for fuel price $1.41/gal
        #TODO: read this froma file
        #TODO: SR-Assumes independednt of load factor -future release will include variation due to load factor
        TotCost_data = 1.0e4*np.array([[[1.0e6],
        	                            [31285.73958],
                                        [1.0E6],
                                        [15437.22222],
                                        [1.0E6],
                                        [13331.18977],
                                        [30717.22701],
                                        [32445.94966],
                                        [22288.5998],
                                        [1.0E6],
                                        [1.0E6]],
                                       [[1.0E6],
                                        [98179.33105],
                                        [124085.0391],
                                        [56591.5614],
                                        [256091.9698],
                                        [51133.53808],
                                        [96676.30095],
                                        [101215.5494],
                                        [74625.81603],
                                        [170126.7763],
                                        [137611.9739]]])

        TotCost_data = np.repeat(TotCost_data,11,axis=2)

        BlockHour_data = np.array([[[25.],
        	                        [5.171178],
                                    [25.],
                                    [2.239887],
                                    [25.],
                                    [1.841388],
                                    [5.067118],
                                    [5.386347],
                                    [3.511025],
                                    [25.0],
                                    [25.0]],
                                   [[25.],
                                    [4.836729028],
                                    [6.543177778],
                                    [2.135766528],
                                    [12.24646217],
                                    [1.77308],
                                    [4.737654139],
                                    [5.03378],
                                    [3.302697917],
                                    [8.648317778],
                                    [7.384417778]]])

        BlockHour_data = np.repeat(BlockHour_data,11,axis=2)

        Fuelburn_data = np.array([[[25.],
        	                        [5.171178],
                                    [25.],
                                    [2.239887],
                                    [25.],
                                    [1.841388],
                                    [5.067118],
                                    [5.386347],
                                    [3.511025],
                                    [25.0],
                                    [25.0]],
                                   [[25.],
                                    [4.836729028],
                                    [6.543177778],
                                    [2.135766528],
                                    [12.24646217],
                                    [1.77308],
                                    [4.737654139],
                                    [5.03378],
                                    [3.302697917],
                                    [8.648317778],
                                    [7.384417778]]])

        Fuelburn_data = np.repeat(Fuelburn_data,11,axis=2)

        # self.AC_name = AC_name[existAC_index]
        self.AC_num = AC_num[existAC_index-1]
        self.seat_cap = seat_cap[existAC_index-1]
        self.des_range = des_range[existAC_index-1]
        self.MH_FH = MH_FH[existAC_index-1]

        #TODO: Perhaps a better way to do this
        for ac in range(len(existAC_index)):
            cost_rt = TotCost_data[existAC_index[ac]-1,route_index-1,:]
            BH_rt = BlockHour_data[existAC_index[ac]-1,route_index-1,:]
            fuelburn_rt = Fuelburn_data[existAC_index[ac]-1,route_index-1,:]
            if ac==0:
                self.TotCost_LF = np.array([cost_rt])
                self.BH_LF = np.array([BH_rt])
                self.fuelburn_LF = np.array([fuelburn_rt])
            else:
                self.BH_LF = np.concatenate((self.BH_LF,np.array([BH_rt])),axis=0)
                self.TotCost_LF = np.concatenate((self.TotCost_LF,np.array([cost_rt])),axis=0)
                self.fuelburn_LF = np.concatenate((self.fuelburn_LF,np.array([fuel_rt])),axis=0)

class NetworkData():
    def __init__(self, num_route, route_index):
        self.num_route = num_route
        # distance_data = np.array([162.0, 753.0, 974.0, 1094.0, 1357.0, 1455.0, 2169.0, 2249.0, 2269, 2337.0, 2350.0])
        # demand_data = np.array([41.0, 1009.0, 89.0, 661.0, 1041.0, 358.0, 146.0, 97.0, 447.0, 194.0, 263.0])
        # price_data = np.array([100.81, 181.59, 219.53, 241.46, 286.43, 303.42, 425.67, 440.53,  444.04, 455.61, 458.31])
        distance_data = np.array([6928.15,	2145.99,	2991.45,	821.69,	5848.02,	641.54,	2098.69,	2241.89,	1391.75,	4039.02,	3407.07])
        demand_data = np.array([10.,	3145.,	4067.,	2237.,	1212.,	57.,	661.,	1808.,	2138.,	329.,	356.])
        price_data = np.array([1431.423766,	442.2878582,	578.350443,	232.9601666,	1204.30695,	205.0699406,	434.7281099,	457.6345773,	323.1007246,	798.6304299,	647.1170824])

        self.route = distance_data[route_index-1]
        self.dem = demand_data[route_index-1]
        self.price = price_data[route_index-1]


class DesAllocFLOPS_1new2ex_11rt(ExternalCode):
    """ 11 route problem with 2 existing aircraft and 1 new yet-to-be-designed aircraft"""
    def __init__(self, inits):
        super(DesAllocFLOPS_1new2ex_11rt, self).__init__()

        #Define the problem here
        self.inits = inits
        # self.num_ac = inits.num_ac
        # self.num_route = inits.num_route
        # self.existAC_index = inits.existAC_index
        # self.route_index = inits.route_index
        # self.num_xC_des = 6 # Number of aircraft design variables - 1.AR 2.TR 3.t2c 4.Area 5.Sweep[deg] 6.ThrustperEngine[lbs]
        # self.alpha = inits.alpha
        # self.beta = inits.beta
        # self.FP = inits.FP
        # self.max_avail_seats = inits.max_avail_seats

        # Continuous Inputs
        self.add_param('xC', np.zeros((self.inits.num_xC_des + 5*self.inits.num_route,)),
                       desc='Continuous type design variables of the des-alloc-rev problem.')

        # Integer Inputs
        self.add_param('xI', np.ones((self.inits.num_ac*self.inits.num_route,)),
                       desc='Integer type design variables of the des-alloc-rev problem')

        # #TODO: In the future release read data from a file
        # # New aircraft
        # self.newac = NewAC(self.route_index)
        # # Existing aircraft
        # self.existac = ExistingAC(self.existAC_index, self.route_index)
        # # Network data
        # self.network = NetworkData(self.num_route,self.route_index)

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_size'] = 1e-3
        self.deriv_options['step_calc'] = 'relative'

        # Outputs
        self.add_output('profit', val=0.0)
        self.add_output('g', val=np.zeros((2+self.inits.num_ac+self.inits.num_route, ))) #TODO:SR- We may not need the 2 aircraft design constraints for AMD

    def solve_nonlinear(self, params, unknowns, resids):
        """ Define the function f(xI, xC)
        Here xI is integer and xC is continuous"""

        xC = params['xC']
        xI = params['xI']

        if MPI.COMM_WORLD.rank == 0:
            # profit, g, g_linineq = obj_cons_calc(xC, xI, self.num_xC_des, self.xClb, self.xCub, self.newac, self.existac, self.network)
            profit, g = obj_cons_calc(xC, xI, self.inits)
        else:
            profit = 0.0
            g = np.zeros((2+self.inits.num_ac + self.inits.num_route,))

        profit = MPI.COMM_WORLD.allgather(profit)[0]
        g = MPI.COMM_WORLD.allgather(g)[0]

        unknowns['profit'] = profit
        unknowns['g'] = g

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


    # Calculate the profit & the constraints
    profit = 0.0
    g = np.zeros((2+num_ac+num_route,))
    #TODO: May ignore this two constraitns-AMD will already have several aircraft design constraints
    g[0] = (TD/8500.0)
    g[1] = (LD/7000.0)
    # Add more performance constraints (like fuselage fuel capacity, landing gear length etc) for the production run
    cc = 1 #Set this to -1 if g[0] and g[1] is not there
    dd = 0
    cost = 0.0

    for kk in range(num_ac):
        con_val = 0.0
        for jj in range(num_route):
            # pax_kj = pax[kk*num_route + jj]
            x_kj = trip[kk*num_route + jj]
            if x_kj > 0:
                if kk == 0: #New aircraft
                    cost_kj = cost_1j[dd]
                    BH_kj = BH_1j[dd]*inits.scale_fac
                    MH_FH_kj = newac.MH_new
                    fuel_kj = fuelburn_1j[dd]
                    dd+=1
                else:
                    LF = int(round(10*pax[kk,jj]/seat[kk]))
                    #TODO #This convention is different in matlab version (3D): dim1-routes,dim2-aircraft, dim3-LF
                    cost_kj = existac.TotCost_LF[kk-1, jj, LF-1]
                    BH_kj = existac.BH_LF[kk-1, jj, LF-1]
                    MH_FH_kj = existac.MH_FH[kk-1]
                    fuel_kj = existac.fuelburn_LF[kk-1, jj, LF-1]
            else:
                cost_kj = 0.0
                BH_kj=0.0
                MH_FH_kj = 0.0

            cost += (cost_kj + fuel_kj*FPperlb)*x_kj
            con_val += x_kj*(BH_kj*(1.0+MH_FH_kj) + 1.0)

        cc += 1
        g[cc] = (con_val/(12*AC_num[kk]))

    for jj in range(num_route):
        cc += 1
        g[cc] = tot_pax[jj]/dem[jj]

    profit = np.sum(rev) - cost

    # print("obj", profit/-1.0e3)
    # print("con", g)
    return profit/-1.0e3, g

def RevenueManagementSys_Det(x1, y1, x2, y2, z1, alpha, beta, Cap):
    # Attractiveness coefficients for fare classes
    a = np.array([0.864, -0.038])
    b = np.array([-0.02, 0.016])
    c = np.array([0.009, 0.008])

    x = np.array([x1,x2])
    y = np.array([y1,y2])

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