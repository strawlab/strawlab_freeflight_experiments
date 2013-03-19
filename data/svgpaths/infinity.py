import numpy as np


N = 100
pi = np.pi
leaft = np.linspace(-pi/4.,pi/4,N)

theta = np.concatenate( (leaft, (leaft+pi)[::-1]))
r = np.cos(2*theta)

x = r*np.cos( theta )
y = r*np.sin( theta )

x *= 150
y *= 280

x += 250
y += 250

if 1:
    TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="500" height="500">
"""
    with open('infinity.svg',mode='w') as fd:
        fd.write(TEMPLATE)
        xy_pairs = []
        for i in range(len(x)):
            xy_pairs.append(( x[i], y[i] ))
        xy_strs = [ '%r,%r'%p for p in xy_pairs ]
        s = 'M ' + ' '.join( xy_strs )
        fd.write('<path d="%s" fill="none" stroke="blue" stroke-width="1"/>'%s)
        fd.write('</svg\n>')

if 1:
    import matplotlib.pyplot as plt

    ax1=plt.subplot(2,1,1,polar=True)
    ax2=plt.subplot(2,1,2)
    ax1.plot( theta, r )
    ax2.plot(x,y)

    plt.show()
