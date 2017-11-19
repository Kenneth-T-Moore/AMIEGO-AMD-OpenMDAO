import numpy as np


##########################
# a/c parameters
##########################

allocation_data = {}
allocation_data['num_existing'] = 2
allocation_data['num_new'] = 1
allocation_data['existing_names'] = ['B738', 'B747']
allocation_data['new_names'] = ['CRM']
allocation_data['names'] = [ 'CRM', 'B738', 'B747']

#
# allocation_data['capacity', 'E170'] = 58
allocation_data['capacity', 'B738'] = 162
# allocation_data['capacity', 'B777'] = 207
allocation_data['capacity', 'B747'] = 416
allocation_data['capacity', 'CRM'] = 300

#
# allocation_data['number', 'E170'] = 100
allocation_data['number', 'B738'] = 18
# allocation_data['number', 'B777'] = 100
allocation_data['number', 'B747'] = 25
allocation_data['number', 'CRM'] = 10

# maintenance hours / block hours
# allocation_data['maint', 'E170'] = 0.936
allocation_data['maint', 'B738'] = 0.948
# allocation_data['maint', 'B777'] = 0.866
allocation_data['maint', 'B747'] = 0.866
allocation_data['maint', 'CRM'] = 0.866

# N
# This factor accounts for the discrepancy between FLOPS and pymission.
factor = 1.5
allocation_data['fuel_N', 'B738'] = np.array([1.00E+15,	27673.51762,	1.00E+15,	11243.88425,	1.00E+15,	9133.7333,	27056.9486,	28928.1346,	18108.09825,	1.00E+15,	1.00E+15]) * 9.81 / 2.2 * factor

allocation_data['fuel_N', 'B747'] = np.array([1.00E+15,	89493.60335,	127293.0433,	36773.28567,	269909.6737,	29913.12705,	87531.91901,	93483.48326,	58966.57353,	176216.4891,	146258.4655]) * 9.81 / 2.2 * factor

allocation_data['block_time_hr', 'B738'] = np.array([1.00E+15,	5.171178,	1.00E+15,	2.239887,	1.00E+15,	1.841388,	5.067118,	5.386347,	3.511025,	1.00E+15,	1.00E+15])

allocation_data['block_time_hr', 'B747'] = np.array([1.00E+15,	4.836729028,	6.543177778,	2.135766528,	12.24646217,	1.77308,	4.737654139,	5.03378,	3.302697917,	8.648317778,	7.384417778])

# $
allocation_data['cost_other', 'B738'] = np.array([1.00E+15,	31285.73958,	9.00E+15,	15437.22222,	9.00E+15,	13331.18977,	30717.22701,	32445.94966,	22288.5998,	9.00E+15,	9.00E+15])
allocation_data['cost_other', 'B747'] = np.array([1.00E+15,	98179.33105,	124085.0391,	56591.5614,	256091.9698,	51133.53808,	96676.30095,	101215.5494,	74625.81603,	170126.7763,	137611.9739])
allocation_data['cost_other', 'CRM'] = np.array([130981.01,	42877.13574,	53463.95286,	24990.15611,	112051.5746,	22593.39838,	42247.3911,	44153.44306,	32840.54423,	74353.72696,	59001.04963])

# $
allocation_data['price_pax', 'B738'] = np.array([0.,	442.2878582,	0.,	232.9601666,	0.,	205.0699406,	434.7281099,	457.6345773,	323.1007246,	0.,	0.])

allocation_data['price_pax', 'B747'] = np.array([0.,	442.2878582,	578.350443,	232.9601666,	1204.30695,	205.0699406,	434.7281099,	457.6345773,	323.1007246,	798.6304299,	647.1170824])
allocation_data['price_pax', 'CRM'] = np.array([1431.423766,	442.2878582,	578.350443,	232.9601666,	1204.30695,	205.0699406,	434.7281099,	457.6345773, 323.1007246,	798.6304299,	647.1170824])


##########################
# route parameters
##########################

allocation_data['num'] = 11

# km
allocation_data['range_km'] = np.array([6928.15,	2145.99,	2991.45,	821.69,	5848.02,	641.54,	2098.69,	2241.89,	1391.75,	4039.02,	3407.07]) * 1.852

#
allocation_data['demand'] = np.array([10.,	3145.,	4067.,	2237.,	1212.,	57.,	661.,	1808.,	2138.,	329.,	356.])

#
allocation_data['num_pt'] = 80 * np.ones(11, int)
allocation_data['num_cp'] = 20 * np.ones(11, int)

# [CRM, 738, 747] in this order (one-way)
allocation_data['flt_day'] = np.array([
    [1,	0,	4,	1,	0,	0,	1,	2,	1,	0,	0],
    [0,	7,	0,	11,	0,	1,	0,	0,	9,	0,	0],
    [0,	5,	7,	1,	3,	0,	1,	3,	1,	1,	1],
])

allocation_data['pax_flt'] = np.array([
    [10,	0,	289,	39,	0,	0,	245,	280,	264,	0,	0],
    [0,	152,	0,	162,	0,	57,	0,	0,	162,	0,	0],
    [0,	416,	416,	416,	404,	0,	416,	416,	416,	329,	356],
])
