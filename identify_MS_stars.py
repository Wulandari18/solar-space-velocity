from lmfit import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.cluster import *
from galpy.potential import *
from galpy.orbit import Orbit
from astropy import units


def Teff(g_bp, g_rp, bp_rp): #ballesteros
    b_v = -30.29578793 * g_bp + 30.19367011 * g_rp - 29.52957408 * bp_rp - 0.0667657186
    T = 4600.0 * ((1.0 / ((0.92 * b_v) + 1.7)) + (1.0 / ((0.92 * b_v) + 0.62)))
    return np.asarray(T)

#identify main sequence and non-main sequence with DBSCAN
g = pd.read_csv('nearby_stars_100pc.csv')
new_g = g.replace([np.inf, -np.inf], np.nan).dropna(axis = 0)
bp_g, g_rp = new_g.bp_g, new_g.g_rp


data = np.column_stack((g_rp, bp_g))
clustering = DBSCAN(eps = 0.02, min_samples = 500).fit(data)
labels = clustering.labels_


label_group1, label_group2 = min(labels), max(labels)
N1, N2 = [0.0 for i in range(2)]

bp_g1, bp_g2, g_rp1, g_rp2 = [[] for i in range(4)]

pmra1, pmra_error1, pmdec1, pmdec_error1, radial_velocity1, radial_velocity_error1 = [[] for i in range(6)]
source_id1, ra1, ra_error1, dec1, dec_error1, parallax1, parallax_error1 = [[] for i in range(7)]

pmra2, pmra_error2, pmdec2, pmdec_error2, radial_velocity2, radial_velocity_error2 = [[] for i in range(6)]
source_id2, ra2, ra_error2, dec2, dec_error2, parallax2, parallax_error2 = [[] for i in range(7)]

l1, b1, l2, b2, bp_rp1, bp_rp2, T1, T2 = [[] for i in range(8)]
for i in range(len(labels)):
    if i%500.0 == 0.0:
        print(i)

    Ti = Teff(-bp_g.tolist()[i], g_rp.tolist()[i], new_g.bp_rp.tolist()[i])
    rai, deci, plxi, pmrai, pmdeci, vri = new_g.ra.tolist()[i], new_g.dec.tolist()[i], new_g.parallax.tolist()[i], new_g.pmra.tolist()[i], new_g.pmdec.tolist()[i], new_g.radial_velocity.tolist()[i]

    if labels[i] == label_group1:
        T1.append(Ti.tolist())
        l1.append(new_g.l.tolist()[i])
        b1.append(new_g.b.tolist()[i])
        bp_rp1.append(new_g.bp_rp.tolist()[i])
        bp_g1.append(bp_g.tolist()[i])
        g_rp1.append(g_rp.tolist()[i])
        source_id1.append(new_g.source_id.tolist()[i])
        ra1.append(new_g.ra.tolist()[i])
        ra_error1.append(new_g.ra_error.tolist()[i])
        dec1.append(new_g.dec.tolist()[i])
        dec_error1.append(new_g.dec_error.tolist()[i])
        pmra1.append(new_g.pmra.tolist()[i])
        pmra_error1.append(new_g.pmra_error.tolist()[i])
        pmdec1.append(new_g.pmdec.tolist()[i])
        pmdec_error1.append(new_g.pmdec_error.tolist()[i])
        radial_velocity1.append(new_g.radial_velocity.tolist()[i])
        radial_velocity_error1.append(new_g.radial_velocity_error.tolist()[i])
        parallax1.append(new_g.parallax.tolist()[i])
        parallax_error1.append(new_g.parallax_error.tolist()[i])
        N1 = N1 + 1

    elif labels[i] == label_group2:
        T2.append(Ti.tolist())
        l2.append(new_g.l.tolist()[i])
        b2.append(new_g.b.tolist()[i])
        bp_rp2.append(new_g.bp_rp.tolist()[i])
        bp_g2.append(bp_g.tolist()[i])
        g_rp2.append(g_rp.tolist()[i])
        source_id2.append(new_g.source_id.tolist()[i])
        ra2.append(new_g.ra.tolist()[i])
        ra_error2.append(new_g.ra_error.tolist()[i])
        dec2.append(new_g.dec.tolist()[i])
        dec_error2.append(new_g.dec_error.tolist()[i])
        pmra2.append(new_g.pmra.tolist()[i])
        pmra_error2.append(new_g.pmra_error.tolist()[i])
        pmdec2.append(new_g.pmdec.tolist()[i])
        pmdec_error2.append(new_g.pmdec_error.tolist()[i])
        radial_velocity2.append(new_g.radial_velocity.tolist()[i])
        radial_velocity_error2.append(new_g.radial_velocity_error.tolist()[i])
        parallax2.append(new_g.parallax.tolist()[i])
        parallax_error2.append(new_g.parallax_error.tolist()[i])
        N2 = N2 + 1

if N1 < N2:
    color1, label1 = 'green', 'non MS'
    color2, label2 = 'red', 'MS'
    np.savetxt('ms_nearby_stars_100pc.txt', np.c_[source_id2, l2, b2, ra2, ra_error2, dec2, dec_error2, parallax2, parallax_error2, pmra2, pmra_error2, pmdec2, pmdec_error2, radial_velocity2, radial_velocity_error2, bp_g2, g_rp2, bp_rp2, T2], header = 'source_id, l, b, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, radial_velocity, radial_velocity_error, bp_g, g_rp, bp_rp, Teff Ballesteros')
    np.savetxt('non-ms_nearby_stars_100pc.txt', np.c_[source_id1, l1, b1, ra1, ra_error1, dec1, dec_error1, parallax1, parallax_error1, pmra1, pmra_error1, pmdec1, pmdec_error1, radial_velocity1, radial_velocity_error1, bp_g1, g_rp1, bp_rp1, T1], header = 'source_id, l, b, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, radial_velocity, radial_velocity_error, bp_g, g_rp, bp_rp, Teff Ballesteros')
else:
    color1, label1 = 'red', 'MS'
    color2, label2 = 'green', 'non MS'
    np.savetxt('ms_nearby_stars_100pc.txt', np.c_[source_id1, l1, b1, ra1, ra_error1, dec1, dec_error1, parallax1, parallax_error1, pmra1, pmra_error1, pmdec1, pmdec_error1, radial_velocity1, radial_velocity_error1, bp_g1, g_rp1, bp_rp1, T1], header = 'source_id, l, b, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, radial_velocity, radial_velocity_error, bp_g, g_rp, bp_rp, Teff Ballesteros')
    np.savetxt('non-ms_nearby_stars_100pc.txt', np.c_[source_id2, l2, b2, ra2, ra_error2, dec2, dec_error2, parallax2, parallax_error2, pmra2, pmra_error2, pmdec2, pmdec_error2, radial_velocity2, radial_velocity_error2, bp_g2, g_rp2, bp_rp2, T2], header = 'source_id, l, b, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, radial_velocity, radial_velocity_error, bp_g, g_rp, bp_rp, Teff Ballesteros')

'''
#active this part for plotting the HR Diagram upon identifying the main sequence
fig = plt.figure(0)
ax1 = fig.add_subplot(111)
ax1.scatter(g_rp1, bp_g1, color = color1, s = 0.1, label = label1)
ax1.scatter(g_rp2, bp_g2, color = color2, s = 0.1, label = label2)
ax1.set_xlabel('$G-R_p$', fontsize = 10)
ax1.set_ylabel('$B_p - G$', fontsize = 10)
ax1.set_ylim(1.75, 0.0)
ax1.set_xlim(0.0, 1.5)
ax1.legend(loc = 'lower left')
fig.savefig('hr.png', dpi = 300)
#end here for DBSCAN
'''

