from lmfit import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ms_nearby_stars_100pc.txt')

source_id, l, b, ra, ra_error, dec, dec_error = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6]
parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error = data[:,7], data[:,8], data[:,9], data[:,10], data[:,11], data[:,12]
radial_velocity, radial_velocity_error, bp_g, g_rp, bp_rp, T_B = data[:,13], data[:,14], data[:,15], data[:,16], data[:,17], data[:,18]

# Select FGK main sequence stars based on Ballesteros
pmra_b, pmra_error_b, pmdec_b, pmdec_error_b, radial_velocity_b, radial_velocity_error_b = [[] for i in range(6)]
source_id_b, ra_b, ra_error_b, dec_b, dec_error_b, parallax_b, parallax_error_b = [[] for i in range(7)]
l_b, b_b, bp_g_b, g_rp_b, bp_rp_b, T_b, type_b = [[] for i in range(7)]

for i in range(len(T_B)):
	if 6000 <= T_B[i] <= 6800:
		type_ = 0				#F
		T_b.append(T_B[i])
		type_b.append(type_)
		source_id_b.append(source_id[i])
		l_b.append(l[i])
		b_b.append(b[i])
		ra_b.append(ra[i])
		ra_error_b.append(ra_error[i])
		dec_b.append(dec[i])
		dec_error_b.append(dec_error[i])
		parallax_b.append(parallax[i])
		parallax_error_b.append(parallax_error[i])
		pmra_b.append(pmra[i])
		pmra_error_b.append(pmra_error[i])
		pmdec_b.append(pmdec[i])
		pmdec_error_b.append(pmdec_error[i])
		radial_velocity_b.append(radial_velocity[i])
		radial_velocity_error_b.append(radial_velocity_error[i])
		bp_g_b.append(bp_g[i])
		g_rp_b.append(g_rp[i])
		bp_rp_b.append(bp_rp[i])
	if 5300 <= T_B[i] < 6000:
		type_ = 1				#G
		T_b.append(T_B[i])
		type_b.append(type_)
		source_id_b.append(source_id[i])
		l_b.append(l[i])
		b_b.append(b[i])
		ra_b.append(ra[i])
		ra_error_b.append(ra_error[i])
		dec_b.append(dec[i])
		dec_error_b.append(dec_error[i])
		parallax_b.append(parallax[i])
		parallax_error_b.append(parallax_error[i])
		pmra_b.append(pmra[i])
		pmra_error_b.append(pmra_error[i])
		pmdec_b.append(pmdec[i])
		pmdec_error_b.append(pmdec_error[i])
		radial_velocity_b.append(radial_velocity[i])
		radial_velocity_error_b.append(radial_velocity_error[i])
		bp_g_b.append(bp_g[i])
		g_rp_b.append(g_rp[i])
		bp_rp_b.append(bp_rp[i])
	elif 4200 <= T_B[i] < 5300:
		type_ = 2				#K
		T_b.append(T_B[i])
		type_b.append(type_)
		source_id_b.append(source_id[i])
		l_b.append(l[i])
		b_b.append(b[i])
		ra_b.append(ra[i])
		ra_error_b.append(ra_error[i])
		dec_b.append(dec[i])
		dec_error_b.append(dec_error[i])
		parallax_b.append(parallax[i])
		parallax_error_b.append(parallax_error[i])
		pmra_b.append(pmra[i])
		pmra_error_b.append(pmra_error[i])
		pmdec_b.append(pmdec[i])
		pmdec_error_b.append(pmdec_error[i])
		radial_velocity_b.append(radial_velocity[i])
		radial_velocity_error_b.append(radial_velocity_error[i])
		bp_g_b.append(bp_g[i])
		g_rp_b.append(g_rp[i])
		bp_rp_b.append(bp_rp[i])

# Calculate stellar velocity
def matrix_B(ra, dec):
	B = []

	for i in range(len(ra)):
		A11 = (np.cos(np.radians(ra[i]))) * (np.cos(np.radians(dec[i])))
		A12 = (np.sin(np.radians(ra[i]))) * (np.cos(np.radians(dec[i])))
		A13 = np.sin(np.radians(dec[i]))
		A21 = -(np.sin(np.radians(ra[i])))
		A22 = np.cos(np.radians(ra[i]))
		A23 = 0
		A31 = -(np.cos(np.radians(ra[i]))) * (np.sin(np.radians(dec[i])))
		A32 = -(np.sin(np.radians(ra[i]))) * (np.sin(np.radians(dec[i])))
		A33 = np.cos(np.radians(dec[i]))

		A = np.array([[A11, A21, A31],
				[A12, A22, A32],
				[A13, A23, A33]])
		
		T = np.array([[-0.06699, -0.87276, -0.48354],
				[0.49273, -0.45035, 0.74458],
				[-0.86760, -0.18837, 0.46020]])

		TA_ = np.matmul(T, A)
		B.append(TA_)
	return B

def matrix_uvw(pmra, par, pmdec, vr, B):
	uvw = []
	for i in range(len(pmra)):
		k = 4.74057
		vt_ra = k*pmra[i]/par[i]
		vt_dec = k*pmdec[i]/par[i]
		v = np.array([[vr[i]], [vt_ra], [vt_dec]])
		uvw_ = np.matmul(B[i], v)
		uvw.append(uvw_)
	return uvw

def matrix_C(B):
	C = []
	for i in range(len(B)):
		C_ = B[i]**2
		C.append(C_)
	return C

def matrix_D(vr_err, par, par_err, pmra, pmra_err, pmdec, pmdec_err):
	D = []
	for i in range(len(par)):
		k = 4.74057
		D11 = vr_err[i]**2
		D12 = ((k/par[i])**2)*((pmra_err[i]**2) + ((pmra[i]*par_err[i]/par[i])**2))
		D13 = ((k/par[i])**2)*((pmdec_err[i]**2) + ((pmdec[i]*par_err[i]/par[i])**2))
		D_ = np.array([[D11], [D12], [D13]])
		D.append(D_)
	return D

def matrix_G(B):
	G = []
	for i in range (len(B)):
		G11 = B[i][0][1] * B[i][0][2]
		G12 = B[i][1][1] * B[i][1][2]
		G13 = B[i][2][1] * B[i][2][2]
		G_ = np.array([[G11], [G12], [G13]])
		G.append(G_)
	return G

def matrix_CD(C, D):
	CD = []
	for i in range(len(C)):
		CD_ = np.matmul(C[i], D[i])
		CD.append(CD_)
	return CD

def matrix_FG(pmra, pmdec, par_err, par, G):
	FG = []
	for i in range(len(pmra)):
		k = 4.74057
		F = 2*pmra[i]*pmdec[i]*(k**2)*(par_err[i]**2)/(par[i]**4)
		FG_ = F*G[i]
		FG.append(FG_)
	return FG

def matrix_uvw_err(CD, FG):
	uvw_err = []
	for i in range(len(CD)):
		err = CD[i] + FG[i]
		uvw_err.append(err)
	return uvw_err

B = matrix_B(ra_b, dec_b)
uvw = matrix_uvw(pmra_b, parallax_b, pmdec_b, radial_velocity_b, B)

U = []
V = []
W = []
for i in range(len(ra_b)):
	u = uvw[i][0]
	v = uvw[i][1]
	w = uvw[i][2]

	U.append(u[0])
	V.append(v[0])
	W.append(w[0])

C = matrix_C(B)
D = matrix_D(radial_velocity_error_b, parallax_b, parallax_error_b, pmra_b, pmra_error_b, pmdec_b, pmdec_error_b)
G = matrix_G(B)
CD = matrix_CD(C, D)
FG = matrix_FG(pmra_b, pmdec_b, parallax_error_b, parallax_b, G)
uvw_err = matrix_uvw_err(CD, FG)

U_err = []
V_err = []
W_err = []
for i in range(len(pmra_b)):
	u_err = uvw_err[i][0]
	v_err = uvw_err[i][1]
	w_err = uvw_err[i][2]

	U_err.append(u_err[0])
	V_err.append(v_err[0])
	W_err.append(w_err[0])

np.savetxt('FGK_ms_nearby_stars_100pc.txt', np.c_[source_id_b, l_b, b_b, ra_b, ra_error_b, dec_b, dec_error_b, parallax_b, parallax_error_b, pmra_b, pmra_error_b, pmdec_b, pmdec_error_b, radial_velocity_b, radial_velocity_error_b, bp_g_b, g_rp_b, bp_rp_b, T_b, type_b, U, U_err, V, V_err, W, W_err], header = 'source_id, l, b, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, radial_velocity, radial_velocity_error, bp_g, g_rp, bp_rp, Teff Ballesteros, type, U, U_error, V, V_error, W, W_error')

# Fitting UVW Histogram
N_u, x_u = np.histogram(U, bins = 'sqrt')
N_v, x_v = np.histogram(V, bins = 'sqrt')
N_w, x_w = np.histogram(W, bins = 'sqrt')

def gaussian(x_dat, mu, sigma, A):
    return A*np.exp(-0.5 * np.square((x_dat - mu)/sigma))


fitting = Model(gaussian)
params_u = fitting.make_params(mu = 0.0, sigma = 0.5, A = 10.0)
params_v = fitting.make_params(mu = 0.0, sigma = 1.0, A = 50.0)
params_w = fitting.make_params(mu = 0.0, sigma = 0.8, A = 40.0)


result_u = fitting.fit(N_u, params_u, x_dat = x_u[:-1])
result_v = fitting.fit(N_v, params_v, x_dat = x_v[:-1])
result_w = fitting.fit(N_w, params_w, x_dat = x_w[:-1])


A_u, mu_u, sigma_u = result_u.best_values.get('A'), result_u.best_values.get('mu'), result_u.best_values.get('sigma')
A_v, mu_v, sigma_v = result_v.best_values.get('A'), result_v.best_values.get('mu'), result_v.best_values.get('sigma')
A_w, mu_w, sigma_w = result_w.best_values.get('A'), result_w.best_values.get('mu'), result_w.best_values.get('sigma')

print('result for U')
print(result_u.fit_report())
print('''
''')
print('result for V')
print(result_v.fit_report())
print('''
''')
print('result for W')
print(result_w.fit_report())
print('''
''')


u_mod, v_mod, w_mod = np.arange(min(x_u[:-1]), max(x_u[:-1]), 0.01), np.arange(min(x_v[:-1]), max(x_v[:-1]), 0.01), np.arange(min(x_w[:-1]), max(x_w[:-1]), 0.01)

fig, ax = plt.subplots(figsize = (20,10), nrows = 3, ncols = 1, sharex = True)

ax[0].hist(U, bins = 'sqrt', histtype = 'step', color = 'red', label = 'U')
ax[0].plot(u_mod, gaussian(u_mod, mu_u, sigma_u, A_u), color = 'black', label = 'U best fit')
ax[0].legend(loc = 'upper left')

ax[1].hist(V, bins = 'sqrt', histtype = 'step', color = 'green', label = 'V')
ax[1].plot(v_mod, gaussian(v_mod, mu_v, sigma_v, A_v), color = 'black', label = 'V best fit')
ax[1].legend(loc = 'upper left')

ax[2].hist(W, bins = 'sqrt', histtype  ='step', color = 'blue', label = 'W')
ax[2].plot(w_mod, gaussian(w_mod, mu_w, sigma_w, A_w), color = 'black', label = 'W best fit')
ax[2].legend(loc = 'upper left')

fig.savefig('UVW_result.png', dpi = 300)

# Calculate Typical Position
Rs = 8.0							#kpc
D_s = []
R_s = []
for i in range(len(parallax_b)):
	D = (1/(parallax_b[i]*(10**-3))) * 10**(-3)		#kpc
	X = Rs - (D*np.cos(np.radians(b_b[i]))*np.cos(np.radians(l_b[i])))
	Y = D*np.cos(np.radians(b_b[i]))*np.sin(np.radians(l_b[i]))
	Z = D*np.sin(np.radians(b_b[i]))
	R = np.sqrt(X**2 + Y**2 + Z**2)
	D_s.append(D)
	R_s.append(R)

N_R, x_R = np.histogram(R_s, bins = 'sqrt')

def gaussian(x_dat, mu, sigma, A):
    return A*np.exp(-0.5 * np.square((x_dat - mu)/sigma))


fitting = Model(gaussian)
params_R = fitting.make_params(mu = 8.5, sigma = 0.5, A = 100.0)

result_R = fitting.fit(N_R, params_R, x_dat = x_R[:-1])

A_R, mu_R, sigma_R = result_R.best_values.get('A'), result_R.best_values.get('mu'), result_R.best_values.get('sigma')


print('result for R')
print(result_R.fit_report())
print('''
''')

'''
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.hist(R_s, bins = 'sqrt', histtype = 'step', color = 'red', label = 'R')
ax1.plot(R_mod, gaussian(R_mod, mu_R, sigma_R, A_R), color = 'black', label = 'R best fit')
ax1.legend(loc = 'upper right')

fig1.savefig('R_result.png', dpi = 300)
'''

# Calculate Solar Peculiar Velocity
Vc = 220			#km/s
Rd = 2.5 			#kpc
R_sigma = 13.70			#kpc
sigma_U = sigma_u
sigma_V = sigma_v
sigma_W = sigma_w
mean_R = mu_R
mean_U = mu_u
mean_V = mu_v
mean_W = mu_w

a = sigma_V**2
b = sigma_W**2
c = mean_R*(sigma_U**2)*((1/Rd)+(2/R_sigma)-(2/mean_R))

Va = (1/(2*Vc))*(a+b+c)

U_sun = -mean_U
V_sun = -Va - mean_V
W_sun = -mean_W

print('Nilai asymmetric drift:', Va, 'km/s')
print('Nilai U_sun:', U_sun, 'km/s')
print('Nilai V_sun:', V_sun, 'km/s')
print('Nilai W_sun:', W_sun, 'km/s')





