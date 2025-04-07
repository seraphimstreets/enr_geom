import numpy as np
def read_arpeggio(fn, ab_chains, ag_chains):
    a = np.zeros((15))
    b = np.zeros((15))
    c = np.zeros((15))
    d = np.zeros((15))
    with open(fn, 'r') as f:
        for line in f.readlines():
            ln = line.split()
            if len(ln) != 18:
                print("?")
                continue
            id1 = ln[0][0]
            id2 = ln[1][0]

            k = ln[2:17]
            k = [int(b) for b in k]
            # k[5] = k[6] = 0
            d += k
            if id1 in ab_chains or id2 in ab_chains:
                a += np.array(k)
            elif id1 in ag_chains or id2 in ag_chains:
                b += np.array(k)
            else:
                c += np.array(k)
    


    return np.concatenate([c,d])
