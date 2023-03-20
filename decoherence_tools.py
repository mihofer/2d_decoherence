import numpy as np
import scipy.integrate
import scipy.special
import scipy.signal
import tfs
import matplotlib.pyplot as plt
import re
import pandas as pd
import multiprocessing

FONTSIZE=15

PEAK_THRESHOLD=1
INTEGRAL_MAX=np.inf


def str_to_jklm(s):
    pattern = re.compile(r"(?P<j>[-]?\d)(?P<k>[-]?\d)(?P<l>[-]?\d)(?P<m>[-]?\d)", re.VERBOSE)
    match = pattern.match(s)
    return int(match.group("j")),int(match.group("k")),int(match.group("l")),int(match.group("m"))
    

def calculate_Ix(I_y, w, tunes, jklm):
    j, k, l,m = str_to_jklm(jklm)
    divisor = ((1-j+k)*tunes['xx'] + (m-l)*tunes['yx']) 
    if divisor==0.:
        divisor=1e-22
    result = 0.5*(w - (1-j+k)*(tunes['x']+tunes['xy']*2*I_y) - (m-l)*(tunes['y']+tunes['yy']*2*I_y))/divisor
#     typo in eq A.20 ?
    if result<0.0:
        return 0.0
    return result


def calculate_Iy(I_x, w, tunes, jklm):
    j, k, l,m = str_to_jklm(jklm)  
    divisor = ((k-j)*tunes['xy'] + (1-l+m)*tunes['yy']) 
    if divisor==0.:
        divisor=1e-22
    result = 0.5*(w - (k-j)*(tunes['x']+tunes['xx']*2*I_x) - (1-l+m)*(tunes['y']+tunes['yx']*2*I_x))/divisor
#     typo in eq A.20 ?
    if result<0.0:
        return 0.0
    return result


def calculate_Ix_limits(w, tunes, jklm):
    j, k, l,m = str_to_jklm(jklm)
#     find Iy where I_{x,mk0}(w, I_y) = offset + slope*I_y > 0
    divisor = ((1-j+k)*tunes['xx'] + (m-l)*tunes['yx'])
    if divisor ==0.:
        divisor=1e-22
    offset = (w - (1-j+k)*tunes['x'] - (m-l)*tunes['y'] )/divisor
    slope = (-(1-j+k)*tunes['xy']*2 - (m-l)*tunes['yy']*2 )/divisor

    if offset >0. and slope ==0.:
        return 0., INTEGRAL_MAX
    if offset <0. and slope <=0.:
        return np.NaN, np.NaN
    if offset >=0. and slope >=0.:
        return 0., INTEGRAL_MAX
    if offset >=0. and slope <0.:
        return 0., np.abs(offset/slope)
    if offset <=0. and slope >0.:
        return np.abs(offset/slope), INTEGRAL_MAX
    
    
def calculate_Iy_limits(w, tunes, jklm):
    j, k, l,m = str_to_jklm(jklm)
#     find Ix where I_{y,mk0}(w, I_x) = offset + slope*I_x > 0
    divisor = ((k-j)*tunes['xy'] + (1-l+m)*tunes['yy'])
    if divisor ==0.:
        divisor=1e-22
    offset = (w - (k-j)*tunes['x'] - (1-l+m)*tunes['y'] )/divisor
    slope = (-(k-j)*tunes['xx']*2 - (1-l+m)*tunes['yx']*2 )/divisor
     
    if offset >0. and slope ==0.:
        return 0., INTEGRAL_MAX
    if offset <0. and slope <=0.:
        return np.NaN, np.NaN
    if offset >=0. and slope >=0.:
        return 0., INTEGRAL_MAX
    if offset >=0. and slope <0.:
        return 0., np.abs(offset/slope)
    if offset <=0. and slope >0.:
        return np.abs(offset/slope), INTEGRAL_MAX


def integrand_Ax(I_y, w, jklm, amplitudes, tunes):
    j, k, l, m = str_to_jklm(jklm)
    I_x = calculate_Ix(I_y, w, tunes, jklm)
    exponent= -0.5*(2*I_x + 2*I_y + amplitudes['x']**2 + amplitudes['y']**2)
    actions_product = (2*I_x)**(0.5*(j+k-1))*(2*I_y)**(0.5*(l+m))
    bessel_functions = scipy.special.iv((1-j+k), amplitudes['x']*np.sqrt(2*I_x))*scipy.special.iv((m-l), amplitudes['y']*np.sqrt(2*I_y))
    return np.exp(exponent)*actions_product*bessel_functions


def integrand_Ay(I_x, w, jklm, amplitudes, tunes):
    j, k, l, m = str_to_jklm(jklm)
    I_y = calculate_Iy(I_x, w, tunes, jklm)
    exponent= -0.5*(2*I_x + 2*I_y + amplitudes['x']**2 + amplitudes['y']**2)
    actions_product = (2*I_x)**(0.5*(j+k))*(2*I_y)**(0.5*(l+m-1))
    bessel_functions = scipy.special.iv((k-j), amplitudes['x']*np.sqrt(2*I_x))*scipy.special.iv((1-l+m), amplitudes['y']*np.sqrt(2*I_y))
    return np.exp(exponent)*actions_product*bessel_functions


def calculate_Ax(w, jklm, amplitudes, tunes):
#     redo this and combine with Ay, separate by jklm/planes and lines
    j, k, l, m = str_to_jklm(jklm)
    if j == 0:
        return 0.0
    
    Imin, Imax = calculate_Ix_limits(w, tunes, jklm)
    if np.NaN in (Imin, Imax):
        return 0.0

    divisor=np.abs((1-j+k)*tunes['xx']+(m-l)*tunes['yx'])
    if divisor==0.:
        divisor=1e-22
    return scipy.integrate.quad(integrand_Ax, Imin, Imax, args=(w, jklm, amplitudes, tunes,), limits=1000)[0] * j/divisor


def calculate_Ay(w, jklm, amplitudes, tunes):
    j, k, l, m = str_to_jklm(jklm)
    if l == 0:
        return 0.0
    
    Imin, Imax = calculate_Iy_limits(w, tunes, jklm)
    
    if np.NaN in (Imin, Imax):
        return 0.0

    divisor=np.abs((k-j)*tunes['xy']+(1-l+m)*tunes['yy'])
    if divisor==0.:
        divisor=1e-22

    return scipy.integrate.quad(integrand_Ay, Imin, Imax, args=(w, jklm, amplitudes, tunes,), limits=1000)[0] * l/divisor


def find_peak_and_width(x,y):
    peaks, _ = scipy.signal.find_peaks(y, prominence=PEAK_THRESHOLD)
    if len(peaks) ==0:
        return np.NaN,np.NaN,0.,0.,[0.,0.]
    results_half = scipy.signal.peak_widths(y, peaks, rel_height=0.5)
    left = x.iloc[round(results_half[2][0])]
    right = x.iloc[round(results_half[3][0])]
    return y.iloc[peaks[0]], x.iloc[peaks[0]], right-left, results_half[1][0], [left, right]

    
def process_df_and_add_spectral_amplitude(df):
    
    jklm = df.headers['jklm']
    amplitudes={'x':df.headers['AX'], 'y':df.headers['AY']}
    tunes={'x':df.headers['QX0'],
           'y':df.headers['QY0'],
           'xx': df.headers['QXX'],
           'xy': df.headers['QXY'],
           'yx': df.headers['QYX'],
           'yy': df.headers['QYY']}

    df['SPECTRAL_AMPLITUDE_X']=df['FREQUENCY'].map(lambda x: calculate_Ax(x, jklm, amplitudes, tunes))
    df['SPECTRAL_AMPLITUDE_Y']=df['FREQUENCY'].map(lambda x: calculate_Ay(x, jklm, amplitudes, tunes))

    df.headers['SPECTRAL_PEAK_X'], df.headers['SPECTRAL_FREQ_PEAK_X'], df.headers['SPECTRAL_WIDTH_X'], _, _ = find_peak_and_width(df['FREQUENCY'], df['SPECTRAL_AMPLITUDE_X'])
    df.headers['SPECTRAL_PEAK_Y'], df.headers['SPECTRAL_FREQ_PEAK_Y'], df.headers['SPECTRAL_WIDTH_Y'], _, _ = find_peak_and_width(df['FREQUENCY'], df['SPECTRAL_AMPLITUDE_Y'])
    
    df.headers['FULL_SPECTRAL_WIDTH_X'] = np.sqrt(return_quadratic_deviation(df['FREQUENCY'], df['SPECTRAL_AMPLITUDE_X']))
    df.headers['FULL_SPECTRAL_WIDTH_Y'] = np.sqrt(return_quadratic_deviation(df['FREQUENCY'], df['SPECTRAL_AMPLITUDE_Y']))
    return df


# create analytical spectrum using RTG thesis 4.15 and 4.16

def Ix_single_plane(w, tunes, jklm):
    j, k, l,m = str_to_jklm(jklm)  
    result = 0.5*(w - (1-j+k)*(tunes['x']))/((1-j+k)*tunes['xx']) 
    if result<0.0:
        return 0.0
    return result    


def analytical_Ax(w, jklm, amplitudes, tunes):
    j, k, l, m = str_to_jklm(jklm)
    if j == 0:
        return 0.0

    I_x = Ix_single_plane(w, tunes, jklm)
    
    exponent= -0.5*(2*I_x + amplitudes['x']**2)
    actions_product = (2*I_x)**(0.5*(j+k-1))
    bessel_functions = scipy.special.iv((1-j+k), amplitudes['x']*np.sqrt(2*I_x))
    return j*np.exp(exponent)*actions_product*bessel_functions/np.abs((1-j+k)*tunes['xx']+(m-l)*tunes['yx'])


def analytical_Ay(freq, jklm, amplitudes, tunes):
    return 0.0


def process_df_and_add_analytical_spectral_amplitude(df):
    
    jklm = df.headers['jklm']
    amplitudes={'x':df.headers['AX'], 'y':df.headers['AY']}
    tunes={'x':df.headers['QX0'],
           'y':df.headers['QY0'],
           'xx': df.headers['QXX'],
           'xy': df.headers['QXY'],
           'yx': df.headers['QYX'],
           'yy': df.headers['QYY']}
    
    df['SPECTRAL_AMPLITUDE_X']=df['FREQUENCY'].map(lambda x: analytical_Ax(x, jklm, amplitudes, tunes))
    df['SPECTRAL_AMPLITUDE_Y']=df['FREQUENCY'].map(lambda x: analytical_Ay(x, jklm, amplitudes, tunes))

    df.headers['SPECTRAL_PEAK_X'], df.headers['SPECTRAL_FREQ_PEAK_X'], df.headers['SPECTRAL_WIDTH_X'], _, _ = find_peak_and_width(df['FREQUENCY'], df['SPECTRAL_AMPLITUDE_X'])
    df.headers['SPECTRAL_PEAK_Y'], df.headers['SPECTRAL_FREQ_PEAK_Y'], df.headers['SPECTRAL_WIDTH_Y'], _, _ = find_peak_and_width(df['FREQUENCY'], df['SPECTRAL_AMPLITUDE_Y'])
    
    df.headers['FULL_SPECTRAL_WIDTH_X'] = np.sqrt(return_quadratic_deviation(df['FREQUENCY'], df['SPECTRAL_AMPLITUDE_X']))
    df.headers['FULL_SPECTRAL_WIDTH_Y'] = np.sqrt(return_quadratic_deviation(df['FREQUENCY'], df['SPECTRAL_AMPLITUDE_Y']))
    return df


def prepare_figure():
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,12))

    ax[0].set_xlabel(r'$w~[2\pi]$', fontsize=FONTSIZE)
    ax[1].set_xlabel(r'$w~[2\pi]$', fontsize=FONTSIZE)
    
    ax[0].tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax[1].tick_params(axis='both', which='major', labelsize=FONTSIZE)

    ax[0].set_ylabel(r'$A_x~[a.u]$', fontsize=FONTSIZE)
    ax[1].set_ylabel(r'$A_y~[a.u]$', fontsize=FONTSIZE)
        
    return fig, ax


def add_spectrum_and_peaks(df, ax, color, label):
    ax[0].plot(df['FREQUENCY'], df['SPECTRAL_AMPLITUDE_X'], color=color, linewidth=2, label=label)
    peak_height, peak_freq, width, height, x_left_and_right = find_peak_and_width(df['FREQUENCY'], df['SPECTRAL_AMPLITUDE_X'])
    if ~np.isnan(peak_freq):
        ax[0].plot(peak_freq, df.loc[df['FREQUENCY'] == peak_freq]['SPECTRAL_AMPLITUDE_X'], color=color, marker='x', markersize=FONTSIZE)
        ax[0].hlines(height, x_left_and_right[0], x_left_and_right[1], color=color, linewidth=2, alpha=0.5)
        ax[0].text(peak_freq, df.loc[df['FREQUENCY'] == peak_freq]['SPECTRAL_AMPLITUDE_X'], f'Peak at {peak_freq:.4e}\nFWHM: {width:.2e}\nsigma: {df.headers["FULL_SPECTRAL_WIDTH_X"]:.2e}', fontsize=FONTSIZE)
    
    ax[1].plot(df['FREQUENCY'], df['SPECTRAL_AMPLITUDE_Y'], color=color, linewidth=2, label=label)
    peak_height,peak_freq, width, height, x_left_and_right = find_peak_and_width(df['FREQUENCY'], df['SPECTRAL_AMPLITUDE_Y'])
    if ~np.isnan(peak_freq):
        ax[1].plot(peak_freq, df.loc[df['FREQUENCY'] == peak_freq]['SPECTRAL_AMPLITUDE_Y'], color=color, marker='x', markersize=5)
        ax[1].hlines(height, x_left_and_right[0], x_left_and_right[1], color=color, linewidth=2)
        ax[1].text(peak_freq, df.loc[df['FREQUENCY'] == peak_freq]['SPECTRAL_AMPLITUDE_Y'], f'Peak at {peak_freq:.4e}\nFWHM: {width:.2e}\nsigma: {df.headers["FULL_SPECTRAL_WIDTH_X"]:.2e}', fontsize=FONTSIZE)
    

def return_quadratic_deviation(x,y):
    w_squared = scipy.integrate.simpson(x**2*y, x)
    w_weighted = scipy.integrate.simpson(x*y, x)
    norm = scipy.integrate.simpson(y, x)
    if norm ==0.:
        return 0.
    return (w_squared/norm)-(w_weighted/norm)**2


def multithreaded_iteration_through_df(df, func):
    pool = multiprocessing.Pool()

    results = pool.map(func, df.to_dict('records'))

    pool.close()
    pool.join()

    return pd.concat(results, ignore_index=True)


def sigma_tune(qxx, ax):
    return 2*qxx*np.sqrt(2+ax**2)


def sigma_sext(qxx, ax):
    return 4*qxx*np.sqrt(3+ax**2)


def sigma_oct(qxx, ax):
    return 6*qxx*np.sqrt(4+ax**2)
