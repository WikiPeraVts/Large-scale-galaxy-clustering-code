'''
Generates the false gaussian distribution as described in the 2015 Farrow paper
(http://arxiv.org/abs/1509.02159) to limit the movement of cloned galaxies to a 
time-independant volume within V_max.
'''

import cython
from libc.stdlib cimport rand, RAND_MAX

def false_gaussian(float x):
    '''
    Approximates to a Gaussian of zero mean, sigma=1/sqrt(12) and peak height of
    unity, but zero beyond abs(x) > 1.0, which corresponds to sqrt(12) * sigma.
    '''
    cdef float absx, md, mx
    absx = abs(x)
    md   = 0.5
    mx   = 1.0
        
    if absx <= md:
        return 1.0 + 6.0 * (absx ** 3 - absx ** 2)
    elif absx <= mx:
        return 2.0 * (1-absx) ** 3
    else:
        return 0.0
        
def refl_win(float a, float b, float lo, float hi, float sig=12.0 ** -0.5):
    '''
    TBD
    '''
    
    cdef float scl = sig * 12.0 ** 0.5
    cdef float xlo = (lo - a) / scl
    cdef float xhi = (hi - a) / scl
    cdef float rng = (hi - lo) / scl
    cdef float x   = (b - a) / scl
    cdef float win, xp, xn
    
    if xhi < x or xlo > x:
        return 0.0
    
    else:
        win = false_gaussian(x)
           
        xp = 2.0 * xhi - x
        while abs(xp) <= 1.0:
            win += false_gaussian(xp)
            if xlo >= xp:
                xp = 2.*xhi-xp
            elif xhi <= xp:
                xp = 2.*xlo-xp
    
        xn = 2.0 * xlo - x
        while abs(xn) <= 1.0:
            win += false_gaussian(xn)
            if xlo >= xn:
                xn = 2.*xhi-xn
            elif xhi <= xn:
                xn = 2.*xlo-xn
        
        return win

def rand_refl_win(float a, float lo, float hi, float sig=12.0 ** -0.5):
    '''
    TBD
    '''
    
    cdef float scl = sig * 12.0 ** 0.5

    cdef float xlo      = (lo - a) / scl
    cdef float xhi      = (hi - a) / scl
    cdef float rng = 2.0
    cdef float sft = 1.0
    cdef float x_guess = 0.0
    cdef float y_guess = 1.0
    cdef float y_win   = 0.5

    while y_guess >= y_win:
        x_guess = rand() / float(RAND_MAX) * rng - sft
        y_guess = rand() / float(RAND_MAX)
        y_win   = false_gaussian(x_guess)
    
    while not(xlo<=x_guess<=xhi):
        if xlo > x_guess:
            x_guess = 2.*xlo-x_guess
        elif xhi < x_guess:
            x_guess = 2.*xhi-x_guess 

    return (x_guess * scl) + a
    
def rrrand():
    cdef float rng = 2.0
    cdef float sft = 1.0
    return rand() / float(RAND_MAX) * rng - sft
