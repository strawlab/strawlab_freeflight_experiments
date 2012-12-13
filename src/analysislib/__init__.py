import numpy as np

def trim_z(allz, minz, maxz):

    valid_cond = (minz < allz) & (allz < maxz)
    valid_z = np.count_nonzero(valid_cond)

    if valid_z == 0:
        return None
    else:
        return valid_cond
