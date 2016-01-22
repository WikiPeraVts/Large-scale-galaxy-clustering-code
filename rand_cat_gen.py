# -*- coding: utf-8 -*-
'''Programme to clone galaxies from a catalogue and generate a random,
unclustered catalogue to examine the effects of clustering on galaxy properties.
'''

import os
import numpy as np
import h5py
import time
import reflwin
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator

__location__ = os.path.realpath(os.path.join(os.getcwd(), \
os.path.dirname(__file__)))

cosmo  = FlatLambdaCDM(H0=100, Om0=0.3)   # , Ode0=0.7
cosmo2 = FlatLambdaCDM(H0=100, Om0=1.0)   # , Ode0=0.0

m_min    = 10.0          # Minimum apparent magnitude usable in the survey
m_max    = 19.8          # Maximum apparent magnitude detectable in the survey
sol_ang  = 0.035752147   # Solid angle over which the first survey is observed
sol_ang2 = 0.040968012   # Solid angle over which the second survey is observed
n_clone  = 20            # The ratio of cloned-to-surveyed galaxies, typ. 400

Phistar = 1.49e-02       # The normalization density for the LF
Alpha   = -1.02          # The power law coefficient for the LF
Magstar = -20.37         # The characteristic luminosity for the LF


data        = h5py.File(os.path.join(__location__,"uniformcat.hdf5"), "r")
Mag         = np.asarray(data["catalogue/mag"][...])
Z           = np.asarray(data["catalogue/z"][...])
Absmag      = np.asarray(data["catalogue/absmag"][...])

data_clus   = h5py.File(os.path.join(__location__,"clusteredcat.hdf5"), "r")
Mag_clus    = np.asarray(data_clus["catalogue/mag"][...])
Z_clus      = np.asarray(data_clus["catalogue/z"][...])
Absmag_clus = np.copy(Mag_clus)
Delta_clus  = np.asarray(data_clus["fluctuations/delta"][...])
Zbin_clus   = np.asarray(data_clus["fluctuations/zbin"][...])


data2       = h5py.File(os.path.join(__location__,"uniformcatwithvmax2.hdf5"), "r")
Mag2        = np.asarray(data2["catalogue/mag"][...])
Z2          = np.asarray(data2["catalogue/z"][...])
Absmag2     = np.asarray(data2["catalogue/absmag"][...])
vmax2       = np.asarray(data2["catalogue/vmax"][...])


k_corr = lambda z: (1.39 * z**2) - (0.75 * z)

# 0.17665 + (0.70855 * z) - (0.064416 * z ** 2) + \
# (4.5458 * z ** 3) + (2.5048 * z ** 4)

abs_mag = lambda z, m, univ: m - univ.distmod(z).value - k_corr(z)
obs_mag = lambda z, M, univ: M + univ.distmod(z).value + k_corr(z)

for i in range(len(Absmag_clus)):
    Absmag_clus[i] = abs_mag(Z_clus[i],Absmag_clus[i],cosmo2)
    
order2  = np.argsort(Absmag2)
Mag2    = Mag2[order2]
Z2      = Z2[order2]
Absmag2 = Absmag2[order2]
vmax2   = np.sort(vmax2)
    

def truncate_by_mag(app_mags, redshifts, abs_mags, vmaxs=None, \
min_app_mag=m_min, vmaxcut=(False,-23,-22.5)):

    mag_run    = np.copy(app_mags)
    z_run      = np.copy(redshifts)
    absmag_run = np.copy(abs_mags) 
    order      = []
    
    if vmaxcut[0] == True:
        
        for i in range(len(abs_mags)):
            if absmag_run[i] < vmaxcut[1] or absmag_run[i] > vmaxcut[2]:
                order.append(i)
        mag_run     = np.delete(mag_run, order)
        z_run       = np.delete(z_run, order)
        absmag_run  = np.delete(absmag_run, order)
        return mag_run, z_run, absmag_run
        
    elif vmaxs == None:
        
        for i in range(len(abs_mags)):
            if mag_run[i] < min_app_mag:
                order.append(i)
        mag_run     = np.delete(mag_run, order)
        z_run       = np.delete(z_run, order)
        absmag_run  = np.delete(absmag_run, order)
        return mag_run, z_run, absmag_run
        
    else:
        
        vmax_run   = np.copy(vmaxs)
    
        for i in range(len(abs_mags)):
            if mag_run[i] < m_min:
                order.append(i)
        mag_run     = np.delete(mag_run, order)
        z_run       = np.delete(z_run, order)
        absmag_run  = np.delete(absmag_run, order)
        vmax_run    = np.delete(vmax_run, order)
        return mag_run, z_run, absmag_run, vmax_run

Mag, Z, Absmag = truncate_by_mag(Mag, Z, Absmag, vmaxcut=(False,-23,-22.5))
Mag_clus, Z_clus, Absmag_clus = truncate_by_mag(Mag_clus, Z_clus, Absmag_clus, \
vmaxcut=(False,-23,-22.5))

#-------------------------------------------------------------------------------

def schechter(magnitude, phiStar=Phistar, alpha=Alpha, MStar=Magstar):
    '''
    Schechter luminosity function by magnitudes.
    '''
    w = 0.4 * (MStar - magnitude)
    return 0.4 * np.log(10) * phiStar * 10.0**(w * (alpha + 1.0)) \
    * np.exp(-10.0**w)
    
    
def redshift_tabulator(univ, N=1e5, solid_Angle=sol_ang):
    '''
    Tables N redshift values between 0 and 1.0, for an AstroPy cosmology univ. 
    Subsquent columns have: the redshift dependant constant 
    q = 5 * (lg(D_L) - 1) + k(z), which is equivalent to m - M; the comoving 
    volume over the solid angle of the surface; the overdensity delta; the 
    change of volume with redshift delta_V; and the sum of all values 
    delta * delta_V from the lowest redshift to the current one.
    '''
    
    intvl = 1.0/N
    table = np.zeros((int(N),6))
    for i in xrange(int(N)):
        z           = intvl + i * intvl
        table[i][0] = z
        table[i][1] = univ.distmod(z).value + k_corr(z)
        table[i][2] = (solid_Angle / (4.0 * np.pi * univ.h ** 3)) \
        * univ.comoving_volume(z).value
        table[i][3] = 1.0
        if i != 0:
            table[i][4] = table[i][2] - table[i-1][2]
            table[i][5] = (table[i][4]*table[i][3]) + table[i-1][5]
        else:
            table[i][4] = 0.0
            table[i][5] = 0.0
    
    return table
    
    
def v_max(table, M):
    '''
    Finds the maximum volume for any given galaxy with absolute magnitude M
    to be translocated and still remain in survey parameters of observed 
    magnitude, using a redshift table.
    '''
    
    q_min = m_min - M
    q_max = m_max - M
    
    index_z_min = np.searchsorted(table[:,1], q_min) # Binary search
    index_z_max = np.searchsorted(table[:,1], q_max)
    
    if table[index_z_max][0] > 0.5:
        index_z_max = np.searchsorted(table[:,0], 0.5)
    
    return table[index_z_max][2] - table[index_z_min][2], \
    table[index_z_max][0], table[index_z_min][0], table[index_z_max][2], \
    table[index_z_min][2]

def v_max_dc(table, M):
    '''
    Finds the maximum, density-corrected volume for any given galaxy with 
    absolute magnitude M to be translocated and still remain in survey 
    parameters of observed magnitude, using a redshift table.
    '''
    
    q_min = m_min - M
    q_max = m_max - M
    
    index_z_min = np.searchsorted(table[:,1], q_min)
    index_z_max = np.searchsorted(table[:,1], q_max)
    
    if table[index_z_max][0] > 0.5:
        index_z_max = np.searchsorted(table[:,0], 0.5)
    
    return table[index_z_max,5] - table[index_z_min,5], \
    table[index_z_max][0], table[index_z_min][0], table[index_z_max][2] * \
    table[index_z_max][3], table[index_z_min][2] * table[index_z_min][3]  
    
    
def lumin_func_est(z, m, table, dM=0.1, absolute=True, density_corr=True):
    '''
    Estimates the luminosity function of galaxies of magnitudes m and redshifts 
    z.
    
    This estimates the luminosity function for a collection of galaxies, given 
    as arrays of corresponding magnitudes and redshifts, over absolute magnitude 
    bins of size dM and using a redshift table.
    
    Parameters
    ----------
    z : array_like
       An array of redshift values.
    m : array_like
       An array of magnitude values, where z[x] and m[x] refer to the same 
       galaxy.
    table : array_like
       A (N,5) array of redshift values and corresponding dependants. The 
       accuracy of the redshift value is 1/N.
    dM : float, optional
       The size of the absolute magnitude bins over which luminosity function
       values are applicable.
    absolute : bool, optional
       True if the magnitudes m are absolute; False if they are apparent.
    density_corr : bool, optional
       True if the density corrected maximum volume is to be used; False if the
       classic density independent value is used.
       
    Returns
    -------
    M_bin : array_like
       An array of the lower limits of the luminosity bins.
    lum_func : array_like
       The estimated luminosity function phi, a function of M_bin values.
    
    '''

    if len(m) != len(z):
        return 'The catalogue must consist of two ordered lists of the ' + \
        'magnitudes m and redshifts z for a catalogue of galaxies of equal size.'
        
    elif absolute == False:
        M = np.zeros(shape=(len(m),))
        for g in xrange(len(m)):
            M[g] = abs_mag(z[g], m[g])
        M_sort = np.sort(M)
    
    elif absolute == True:
        M_sort = np.sort(m)
    
    run_M    = min(M_sort)
    max_M    = max(M_sort)
    i        = 0
    lum_func = []
    M_bin    = []
    
    if density_corr == True:
        while run_M < max_M:
            Sum  = 0.0
            while i < len(M_sort) and M_sort[i] < run_M + dM:
                Sum += 1.0 / v_max_dc(table, M_sort[i])[0]
                i   += 1
            lum_func.append(Sum/dM)
            M_bin.append(run_M)
            run_M += dM
    elif density_corr == False:
        while run_M < max_M:
            Sum  = 0.0
            while i < len(M_sort) and M_sort[i] < run_M + dM:
                Sum += 1.0 / v_max(table, M_sort[i])[0]
                i   += 1
            lum_func.append(Sum/dM)
            M_bin.append(run_M)
            run_M += dM
    elif density_corr == 'Analytic':
        while run_M < max_M:
            Sum  = 0.0
            while i < len(M_sort) and M_sort[i] < run_M + dM:
                Sum += 1.0 / vmax2[i]
                i   += 1
            lum_func.append(Sum/dM)
            M_bin.append(run_M)
            run_M += dM
        
    return M_bin, lum_func
    

#-------------------------------------------------------------------------------


def rand_cat_populator(z, M, table, attributes, windowed=False):
    '''
    Populatates a random catalogue with clones from an initial catalogue (z, M,
    attributes), using a redshift table defined by redshift_tabulator(cosmo).
    '''
    
    rand_cat = []
    n_list   = []
    v_list   = []
    vdc_list = []
    Mu       = 0.0
    
    if windowed == False:
           
        for i in xrange(len(z)):
            v_list.append(v_max(table, M[i]))
            vdc_list.append(v_max_dc(table, M[i]))
        
        v_list   = np.asarray(v_list)
        vdc_list = np.asarray(vdc_list)
            
        for i in xrange(len(z)):
            v, z_mx, z_mn, v_mx, v_mn                = v_list[i]
            v_dc, z_mx_dc, z_mn_dc, v_mx_dc, v_mn_dc = vdc_list[i]
            n                                        = n_clone * v / v_dc        
            n_list.append(n)
            
            prob_n = 1 - (n % 1)
            lot_n  = np.random.rand()
            
            if lot_n <= prob_n:
                n = np.floor(n)
            else:
                n = np.ceil(n)            
        
            for j in xrange(int(n)):
                new_v = v_mn + np.random.random() * v
                new_z = table[np.searchsorted(table[:,2], new_v)][0]
                while new_z > 0.5:
                    new_v = v_mn + np.random.random() * v
                    new_z = table[np.searchsorted(table[:,2], new_v)][0]
                rand_cat.append([new_z, M[i], attributes[i]])
        
        return np.asarray(rand_cat), n_list
        
    elif windowed == True:
           
        for i in xrange(len(z)):
            v, z_mx, z_mn, v_mx, v_mn                = v_max(table, M[i])
            v_dc, z_mx_dc, z_mn_dc, v_mx_dc, v_mn_dc = v_max_dc(table, M[i])
            n                                        = n_clone * v / v_dc
            centre                                   = v_mn + (v_mx-v_mn) / 2.0
            n_list.append(n)
            
            prob_n = 1 - (n % 1)
            lot_n  = np.random.rand()
            
            if lot_n <= prob_n:
                n = np.floor(n)
            else:
                n = np.ceil(n)
        
            for j in xrange(int(n)):
                new_v = v_mn + reflwin.rand_refl_win(centre,v_mn,v_mx) * \
                (v_mx-v_mn)
                new_z = table[np.searchsorted(table[:,2], new_v)][0]
                while new_z > 0.5:
                    new_v = v_mn + reflwin.rand_refl_win(centre,v_mn,v_mx) * \
                    (v_mx-v_mn)
                    new_z = table[np.searchsorted(table[:,2], new_v)][0]
                rand_cat.append([new_z, M[i], attributes[i]])
        
        return np.asarray(rand_cat), n_list
    
    
def overdensity_iter(z, M, table, attributes, N, dz=0.025):
    '''
    Runs rand_cat_populator for N iterations, correcting the overdensity array 
    after each run.
    '''
    
    run_table = table.copy()
    delta_tab = []
    table_rec = []
    
    for i in xrange(N):
        
        table_rec.append(run_table)
        run_cat  = rand_cat_populator(z, M, table_rec[-1], attributes)[0]
        z_r      = run_cat[:,0]
               
        z_sort   = np.sort(z)
        z_r_sort = np.sort(z_r)
           
        run_z    = 0.0125    # min(run_table[:,0])
        max_z    = 0.48751   # max(run_table[:,0])
    
        while run_z < max_z:
            
            i_r_min = np.searchsorted(z_r_sort, run_z)
            i_r_max = np.searchsorted(z_r_sort, run_z+dz)
            i_g_min = np.searchsorted(z_sort, run_z)
            i_g_max = np.searchsorted(z_sort, run_z+dz)
            
            n_r = i_r_max - i_r_min
            n_g = i_g_max - i_g_min
            
            if n_r != 0:
                delta_run = n_clone * float(n_g) / float(n_r)
            else:
                delta_run = 0.0
            delta_tab.append(delta_run)
            
            j = np.searchsorted(run_table[:,0], run_z)
            j_max = np.searchsorted(run_table[:,0], run_z+dz)
            
            while j < j_max:
                
                run_table[j][3] = delta_run
                
                if j != 0:
                    
                    run_table[j][5] = (run_table[j][4]*delta_run) + \
                    run_table[j-1][5]
                    
                else:
                    
                    run_table[j][5] = 0.0
                j += 1
            
            run_z += dz
            
        run_table = run_table.copy()


    return run_table, run_cat, delta_tab, table_rec
    
    
    
def overdensity_plot(actual, actual_bin, deltalist, deltalist_bin, N):

    pyplot.plot(actual_bin, actual, '-', color='k')

    grouplen = len(deltalist) / N
    run      = 0
    for i in xrange(N):
        pyplot.plot(deltalist_bin, deltalist[run:run+grouplen], '-')
        run += grouplen   

    pyplot.xlabel(r'$z$', fontsize=48)
    pyplot.ylabel(r'$\Delta$', fontsize=48)

    pyplot.show()




#-------------------------------------------------------------------------------


Table_test  = redshift_tabulator(cosmo)
Table_test2 = redshift_tabulator(cosmo2,solid_Angle=sol_ang2)
Table_test3 = redshift_tabulator(cosmo,solid_Angle=sol_ang2)

def test(N=200000, clustered=False, build_new_table=(False, Table_test)):
    '''
    Test of the code with a uniform catalogue. The automatic number of galaxies
    (N=200000) and a table of 10e6 redshifts will take ~10 minutes currently.
    - 12/11/15
    '''

    log_phi  = []
    log_phi2 = []
    log_phi3 = []

    a          = time.time()
    if build_new_table[0] == True:
        table_test = redshift_tabulator(cosmo)
    elif build_new_table[0] == False:
        table_test = build_new_table[1]
    b          = time.time()

    if clustered == False:
        Z_test = np.copy(Z)           
        Absmag_test = np.copy(Absmag)

    elif clustered == True:
        Z_test = np.copy(Z_clus)
        Absmag_test = np.copy(Absmag_clus)

    d            = time.time()
    M_bin, phi   = lumin_func_est(Z_test[:N-1], Absmag_test[:N-1], table_test, \
    density_corr=False)
    M_bin2, phi2 = lumin_func_est(Z_test[:N-1], Absmag_test[:N-1], table_test)
#    M_bin3, phi3 = lumin_func_est(Z2, Absmag2, table_test, \
#    density_corr='Analytic')  
    e            = time.time()

    for i in xrange(len(phi)):

        if isinstance(phi[i], u.quantity.Quantity):
            log_phi.append(np.log10(phi[i].value/2.0))
            log_phi2.append(np.log10(phi2[i].value/2.0))
 #           log_phi3.append(np.log10(phi3[i].value/2.0))

        else:    
            log_phi.append(np.log10(phi[i]/2.0))
            log_phi2.append(np.log10(phi2[i]/2.0))
#            log_phi3.append(np.log10(phi3[i]/2.0))

    log_phi  = np.asarray(log_phi)
    log_phi2 = np.asarray(log_phi2)
#    log_phi3 = np.asarray(log_phi3)

    if len(M_bin) != len(log_phi):
        return len(M_bin), len(log_phi)
        
    pyplot.figure()

    pyplot.xlabel(r'$M - 5 \lg h$', fontsize=24)
    pyplot.ylabel(\
    r'$\lg (\phi(M) / \frac{h}{\mathrm{Mpc}}^{3} \mathrm{mag^{-1}}))$', \
    fontsize=24)

    pyplot.plot(M_bin - 5 * np.log10(cosmo.h), log_phi, '-', color='b')
    pyplot.plot(M_bin2 - 5 * np.log10(cosmo.h), log_phi2, '-', color='r')
    pyplot.plot(M_bin - 5 * np.log10(cosmo.h), \
    [np.log10(schechter(M_bin[i])) for i in range(len(M_bin))], '--', \
    color='k')
#    pyplot.plot(M_bin3 - 5 * np.log10(cosmo.h), log_phi3, '-', color='k')

    pyplot.gca().invert_xaxis()
    pyplot.show()

    return b - a, e - d, M_bin, phi, log_phi


table_build_time, lumin_func_est_time, M_bin, phi, log_phi = test()


#-------------------------------------------------------------------------------


def magnitude():
    pyplot.subplot(121)
    pyplot.title('uniformcat.hdf5')
    n, bins, patches = pyplot.hist(Mag, 100, normed=0, facecolor='green', \
    alpha=0.8)

    pyplot.xlabel('$m$', fontsize=24)
    pyplot.ylabel('$N$', fontsize=24)

    a = pyplot.axes([0.18, .3, .2, .3])
    pyplot.hist(Mag, 100, cumulative=1, facecolor='red', alpha=0.8)
    a.set_ylabel(r'$F(N)$')
    a.set_yscale('log')
    #pyplot.axis([40, 160, 0, 0.03])

    pyplot.subplot(122)
    pyplot.title('clusteredcat.hdf5')
    n2, bins2, patches2 = pyplot.hist(Mag_clus, 100, normed=0, \
    facecolor='green', alpha=0.8)

    pyplot.xlabel('$m$', fontsize=24)
    pyplot.ylabel('$N$', fontsize=24)

    b = pyplot.axes([.6, .3, .2, .3])
    pyplot.hist(Mag_clus, 100, cumulative=1, facecolor='red', alpha=0.8)
    b.set_ylabel(r'$F(N)$')
    b.set_yscale('log')

    pyplot.show()


    
#def redshift():
#    ax = pyplot.subplot(121)
#    pyplot.title('uniformcat.hdf5')
#    n, bins, patches = pyplot.hist(Z, 100, normed=0, facecolor='red', alpha=0.8)
#    minorLocator = MultipleLocator(0.02)
#
#    pyplot.xlabel('$z$', fontsize=24)
#    pyplot.ylabel('$N$', fontsize=24)
#    ax.xaxis.set_minor_locator(minorLocator)
#
#    pyplot.subplot(122)
#    pyplot.title('clusteredcat.hdf5')
#    n2, bins2, patches2 = pyplot.hist(Z_clus, 100, normed=0, facecolor='red', \
#    alpha=0.8)
#
#    pyplot.xlabel('$z$', fontsize=24)
#    pyplot.ylabel('$N$', fontsize=24)
#    ax.xaxis.set_minor_locator(minorLocator)
#
#    pyplot.show()

def redshift():
    ax=pyplot.subplot(111)
    n, bins, patches = pyplot.hist(Z, 100, normed=0, \
    histtype='step', facecolor='red', alpha=0.0)
    bincenters       = 0.5*(bins[1:]+bins[:-1])
    pyplot.plot(bincenters,n,color='red')
    pyplot.plot(0,0,color='black')
    n2, bins2, patches2 = pyplot.hist(Z_clus, 100, normed=0, histtype='step', \
    color='black',     alpha=1.0)
    
    pyplot.xlabel('$z$', fontsize=40)
    pyplot.ylabel('$N$', fontsize=40)
    minorLocator = MultipleLocator(0.02)
    minorLocator2 = MultipleLocator(100)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator2)
    ax.xaxis.set_tick_params(length=8, width=1, labelsize=30)
    ax.yaxis.set_tick_params(length=8, width=1, labelsize=30)
    ax.xaxis.set_tick_params(which=u'minor',length=4, width=1)
    ax.yaxis.set_tick_params(which=u'minor',length=4, width=1)
    pyplot.legend(['uniformcat.hdf5', 'clusteredcat.hdf5'], fontsize=30, \
    framealpha=0.0)
    pyplot.ylim(7, 4000)
    

    pyplot.show()

def absolute_magnitude():
    pyplot.subplot(121)
    pyplot.title('uniformcat.hdf5')
    n, bins, patches = pyplot.hist(Absmag, 100, normed=0, facecolor='blue', \
    alpha=0.8)

    pyplot.xlabel('$M$', fontsize=24)
    pyplot.ylabel('$N$', fontsize=24)

    a = pyplot.axes([0.3, .4, .15, .15])
    pyplot.hist(np.sort(Absmag)[-40:], 20, cumulative=0, facecolor='blue', \
    alpha=0.8)
    a.set_ylabel(r'$N$')
    #pyplot.axis([40, 160, 0, 0.03])

    pyplot.subplot(122)
    pyplot.title('clusteredcat.hdf5')
    n2, bins2, patches2 = pyplot.hist(Absmag_clus, 100, normed=0, \
    facecolor='blue', alpha=0.8)

    pyplot.xlabel('$M$', fontsize=24)
    pyplot.ylabel('$N$', fontsize=24)

    b = pyplot.axes([.72, .4, .15, .15])
    pyplot.hist(np.sort(Absmag_clus)[-40:], 20, cumulative=0, \
    facecolor='blue', alpha=0.8)
    b.set_ylabel(r'$N$')

    pyplot.show()

    
def M_Z():
    pyplot.subplot(121)
    pyplot.title('uniformcat.hdf5')
    pyplot.plot(Absmag, Z, 'x', color='k')

    pyplot.xlabel('$M$', fontsize=24)
    pyplot.ylabel('$z$', fontsize=24)

    pyplot.subplot(122)
    pyplot.title('clusteredcat.hdf5')
    pyplot.plot(Absmag_clus, Z_clus, 'x', color='k')

    pyplot.xlabel('$M$', fontsize=24)
    pyplot.ylabel('$z$', fontsize=24)

    pyplot.show()    
