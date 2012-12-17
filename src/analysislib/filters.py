import numpy as np

FILTER_REMOVE = "remove"
FILTER_TRIM   = "trim"
FILTER_NOOP   = "none"

def filter_z(method, allz, minz, maxz):
    """
    returns a boolean ndarray that can be used to index trajectory arrays to
    only return values according to this filter

    REMOVE: remove all values outside the Z-range, can cause 'holes'
            in trajectory data
    TRIM:   remove all values after the first time the object leaves the
            valid zone.
    NOOP:   remove no values
    """

    if method == FILTER_NOOP:
        return np.ones_like(allz, dtype=np.bool)

    #select values inside the range
    cond = (minz < allz) & (allz < maxz)

    if method == FILTER_REMOVE:
        return cond
    elif method == FILTER_TRIM:
        #stop considering trajectory from the moment it leaves valid zone
        bad_idxs = np.nonzero(~cond)[0]
        if len(bad_idxs):
            cond = np.ones_like(allz, dtype=np.bool)
            cond[bad_idxs[0]:] = False
            return cond
        else:
            #keep all data
            return np.ones_like(allz, dtype=np.bool)
    else:
        raise Exception("Unknown filter method")

def filter_radius(method, allx, ally, maxr):
    """
    returns a boolean ndarray that can be used to index trajectory arrays to
    only return values according to this filter

    REMOVE: remove all values outside the Z-range, can cause 'holes'
            in trajectory data
    TRIM:   remove all values after the first time the object leaves the
            valid zone.
    NOOP:   remove no values
    """

    if method == FILTER_NOOP:
        return np.ones_like(allx, dtype=np.bool)

    #select values inside the range
    rad = np.sqrt(allx**2 + ally**2)
    cond = rad < maxr

    if method == FILTER_REMOVE:
        return cond
    elif method == FILTER_TRIM:
        #stop considering trajectory from the moment it leaves valid zone
        bad_idxs = np.nonzero(~cond)[0]
        if len(bad_idxs):
            cond = np.ones_like(allx, dtype=np.bool)
            cond[bad_idxs[0]:] = False
            return cond
        else:
            #keep all data
            return np.ones_like(allx, dtype=np.bool)
    else:
        raise Exception("Unknown filter method")

if __name__ == "__main__":
    z = np.array([2,2,3,2,3,4,5,5,4,3,2,2,0])

    print "z     ", z
    print "remove", z[filter_z(FILTER_REMOVE,z,1,4)]
    print "trim  ", z[filter_z(FILTER_TRIM,z,1,4)]
    print "noop  ", z[filter_z(FILTER_NOOP,z,1,4)]

    x = np.array([2,2,2,2,2,2,2])
    y = np.array([2,2,2,3,4,2,4])
    print "y     ", y
    print "remove", y[filter_radius(FILTER_REMOVE,x,y,4.2)]
    print "trim  ", y[filter_radius(FILTER_TRIM,x,y,4.2)]
    print "noop  ", y[filter_radius(FILTER_NOOP,x,y,4.2)]


