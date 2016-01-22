'''
Generates the false gaussian distribution as described in the 2015 Farrow paper
(http://arxiv.org/abs/1509.02159) to limit the movement of cloned galaxies to a 
time-independant volume.
'''

import numpy as np


def false_gaussian(x):
    '''
    Approximates to a Gaussian of zero mean, sigma=1/sqrt(12) and peak height of
    unity, but zero beyond abs(x) > 1.0, which corresponds to sqrt(12) * sigma.
    '''
    absx = abs(x)
    
    if absx <= 0.5:
        return 1.0 + 6.0 * (absx ** 3 - absx ** 2)
    elif absx <= 1.0:
        return 2.0 * (1-absx) ** 3
    else:
        return 0.0

def reflected_window(a, b, lo, hi, sig=12.0 ** -0.5):
    '''
    TBD
    '''
    
    scl = (2 * np.sqrt(3) * sig) ** -1
    xlo = (lo - a) * scl
    xhi = (hi - a) * scl
    x   = (b - a) * scl
    
    if xhi < x or xlo > x:
        return 0.0
    
    else:
        win = false_gaussian(x)
           
        xp = 2.0 * xhi - x
        while xp <= 1.0:
            win = win + false_gaussian(xp)
            xp  = xp + (xhi - xlo)
    
        xn = 2.0 * xlo - x
        while xn >= -1.0:
            win = win + false_gaussian(xn)
            xn  = xn - (xhi - xlo)
        
        return win
    
    
def rand_refl_win(a, lo, hi, sig=12.0 ** -0.5):
    '''
    TBD
    '''
    
    width = hi - lo
    scl   = (2 * np.sqrt(3) * sig) ** -1
    
    def peak_height():
        if hi-a > np.sqrt(3)*sig and lo-a < -np.sqrt(3)*sig:
            return 1.0
        elif bool(hi-a > np.sqrt(3)*sig) is not bool(lo-a < -np.sqrt(3)*sig):
            return 2.0
        else:
            return 2.7 * 1.5 * np.sqrt(3) * sig / width
    
    height  = peak_height()
    mn      = max(lo-a,-scl)
    mx      = min(hi-a,scl)
    rnge    = mx - mn
    x_guess = a
    y_guess = 1.0
    y_win   = 0.5
    
    while y_guess > y_win:
        x_guess = np.random.rand() * rnge + (a + mn)
        y_guess = np.random.rand() * height
        y_win   = reflected_window(a, x_guess, lo, hi, sig)
        
    return x_guess
    
    
def win_test():
    lo  = -0.001
    hi  = 0.001
    a   = 1.0
    sig = 12.0 ** -0.5

    Sum = 0.0
    bmin = max(lo, a-3.47*sig)
    bmax = min(hi, a+3.47*sig)
    db   = (bmax - bmin) / 500.0
    for i in np.arange(bmin, bmax, db):
        Sum += reflected_window(a, i, lo, hi, sig) * db
        
    return 'The numerical solution is ' + str(Sum) + '. The ratio of ' \
    'numerical and analytic solutions is ' + str(Sum/0.75)