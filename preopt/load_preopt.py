from six import iteritems

import numpy as np


int_con = [	'allocation_mission_group.allocation_group.profit_comp.g_aircraft_new',
                   'allocation_mission_group.allocation_group.profit_comp.g_aircraft_exist',
                   'allocation_mission_group.allocation_group.profit_comp.g_demand']

def parse_preopts(raw):
    """
    Read in points from saved run.
    """
    cons = {}
    lb = {}
    ub = {}
    objs = {'profit' : []}

    with open(raw, 'r') as f:
        lines = f.readlines()

    # -----------
    # SNOPT point
    # -----------

    # Constraints
    for j, line in enumerate(lines):
        if not line.startswith('        Name    Type                    Bounds'):
            continue
        break

    lines = lines[j+1:]
    for j, line in enumerate(lines):
        if line.startswith('-----'):
            break
        name, _, low, _, val, _, high = line.split()

        if name not in int_con:
            continue

        if name not in cons:
            cons[name] = []
            lb[name] = []
            ub[name] = []

        cons[name].append(float(val))
        lb[name].append(float(low))
        ub[name].append(float(high))

    # Turn lists into arrays:
    for name, val in iteritems(cons):
        cons[name] = []
        cons[name].append(np.atleast_1d(val))

    # Objective
    lines = lines[j+4:]
    objs['profit'].append(np.atleast_1d(float(lines[0].split()[1])))

    # ----------------
    # Screened Preopts
    # ----------------

    for k in range(66):
        for j, line in enumerate(lines):
            if not line.startswith('profit ['):
                continue
            break

        # Profit is easy
        lines = lines[j:]
        profit = np.atleast_1d(float(lines[0].replace('profit [', '').replace(']', '')))
        objs['profit'].append(profit)

        # Cons are a mess
        lines = lines[1:]
        current = None
        vals =  []
        end = False
        for j, line in enumerate(lines):
            if line.startswith('Exit Flag'):
                break

            # Clean up any junk
            line = line.replace('cons {', '')

            words = line.split()

            for word in words:

                if '}' in word:
                    end = True

                if "'" in word:
                    current = word.replace("'", "").replace(":", "")
                    vals =  []
                elif word in ['array([', ',']:
                    continue
                elif ']' in word:
                    word = word.replace('])', '').replace('array([', '').replace('}', '')
                    if word not in [',', '']:
                        vals.append(float(word.replace(',', '')))

                    if current not in int_con:
                        continue

                    cons[current].append(np.atleast_1d(vals))
                    current = None
                else:
                    vals.append(float(word.replace(',', '')))

            if end:
                break

        lines = lines[j:]

    # Sanity check
    for key in cons:
        n1 = len(cons[key][0])
        n2 = len(cons[key][1])
        if n1 != n2:
            print("Error in", key, n1, n2)

    # Save to numpy

    return objs, cons, lb, ub

def check_surrogate(objs, cons, lb, ub):
    r_pen = 2.0

    x_i = []
    obj_surr = objs['profit']
    n = len(obj_surr)
    P = np.zeros((n, 1))
    num_vio = np.zeros((n, 1), dtype=np.int)

    print('Profit')
    print(obj_surr)

    # Normalize the objective data
    #X_mean = np.mean(x_i, axis=0)
    # X_std = np.std(x_i, axis=0)
    # X_std[X_std == 0.] = 1.

    Y_mean = np.mean(obj_surr, axis=0)
    Y_std = np.std(obj_surr, axis=0)
    Y_std[Y_std == 0.] = 1.

    #X = (x_i - X_mean) / X_std
    Y = (obj_surr - Y_mean) / Y_std

    print('Normalized Profit (Y)')
    print(Y)

    for name, val in iteritems(cons):

        if name not in int_con:
            continue

        val = np.atleast_1d(val)

        meta = {}

        # Note, Branch and Bound defines constraints to be violated
        # when positive, so we need to transform from OpenMDAO's
        # freeform.
        val_u = val - ub[name]
        val_l = lb[name] - val

        # Normalize the constraint data
        g_mean = np.mean(val, axis=0)
        g_std = np.std(val, axis=0)
        g_std[g_std == 0.] = 1.0
        g_norm = (val - g_mean) / g_std
        g_vio_ub = val_u / g_std
        g_vio_lb = val_l / g_std

        # Make the problem appear unconstrained to Amiego
        M = val.shape[1]
        for ii in range(n):
            for mm in range(M):

                if val_u[ii][mm] > 0:
                    P[ii] += g_vio_ub[ii][mm]**2
                    num_vio[ii] += 1

                elif val_l[ii][mm] > 0:
                    P[ii] += g_vio_lb[ii][mm]**2
                    num_vio[ii] += 1

    for ii in range(n):
        if num_vio[ii] > 0:
            Y[ii] += (r_pen * P[ii] / num_vio[ii])

    print('Penalized Profit (Y + sum((r_pen * P[ii] / num_vio[ii])))')
    print(Y)

def load_all_preopts(raw):
    objs, old_cons, lb, ub = parse_preopts(raw)

    cons = {}
    for key in int_con:
        new_key = key.split('.')[-1]
        cons[new_key] = old_cons[key]

    eflag = np.zeros(len(objs['profit']))
    eflag[0] = 1.0
    return objs, cons, list(eflag)


if __name__ == "__main__":

    raw = "raw"
    objs, cons, lb, ub = parse_preopts(raw)
    check_surrogate(objs, cons, lb, ub)

    obj, con, eflag = load_all_preopts(raw)
    print(obj, con, eflag)
    print('done')
