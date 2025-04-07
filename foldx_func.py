
import numpy as np
def foldx_calc(fn, chainid, extra_chains):
    feat = np.zeros((29,))
    feat2 = np.zeros(29,)
    with open(fn, "r") as f:
        lines = f.readlines()[9:]
        for ln in lines:
            pts = ln.split()
            c1, c2 = pts[1], pts[2]
            x = pts[3:]
            x2 = np.array([float(a) for a in x])
            # if c1 == chainid or c2 == chainid:
            #     feat += x2
            # if c1 in extra_chains or c2 in extra_chains:
            #     feat2 += x2
            feat += x2


    return np.concatenate([feat])