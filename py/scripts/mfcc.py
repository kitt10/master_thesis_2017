#!/usr/bin/env python
# -*- coding: cp1250 -*-

from argparse import ArgumentParser
from scipy.io.wavfile import read as read_wav
from scipy.fftpack import dct as scipy_dct
from numpy import hamming, dot, multiply, fft, log10, power, array
from matplotlib import pyplot as plt, rcParams as mpl_params

mpl_params['axes.labelsize'] = 22
mpl_params['xtick.labelsize'] = 20
mpl_params['ytick.labelsize'] = 20


def parse_arguments():
    parser = ArgumentParser(description='MFCC Audio Data Parametrizer.')
    parser.add_argument('-in', '--input_file', type=str, required=True,
                        help='Path to the input .wav file.')
    parser.add_argument('-ms', '--microsegment_dur', type=float, default=0.032, required=False,
                        help='Microsegment duration [s].')
    parser.add_argument('-sh', '--shift_dur', type=float, default=0.01, required=False,
                        help='Shift duration [s].')
    parser.add_argument('-fil', '--show_filters', type=bool, default=False,
                        help='Show MFCC filters?')
    parser.add_argument('-ver', '--verbose', type=bool, default=True,
                        help='Print analysis and plot coefficients?')
    return parser.parse_args()


def normalize(s):
    return [(float(s_i)/(2**15.)) for s_i in s]


def hz2mel(f):
    return 2595*log10(1+(f/700.0))


def mel2hz(f):
    return 700*(power(10, (f/2595.0))-1)


def get_filter(f_def, bi, bm_):
    the_filter = list()
    for i, f in enumerate(f_def):
        if f < bm_[bi-1] or f >= bm_[bi+1]:
            the_filter.append(0.0)
        elif bm_[bi-1] <= f < bm_[bi]:
            the_filter.append(float((f-bm_[bi-1]))/float((bm_[bi]-bm_[bi-1])))
        elif bm_[bi] <= f < bm_[bi+1]:
            the_filter.append(float((f-bm_[bi+1]))/float((bm_[bi]-bm_[bi+1])))
    return array(the_filter)


def design_filters(show_filters=False):
    bm_ = [0.0]
    for i in range(n_filters+1):
        bm_.append(bm_[-1]+db_m)
    
    filters = list()
    f_def_m = [i * f_def_step_m for i in range(filter_len+1)]
    for bi in range(1, n_filters+1):
        filters.append(get_filter(f_def=f_def_m, bi=bi, bm_=bm_))

    if show_filters:
        plt.clf()
        for a_filter in filters:
            plt.plot(mel2hz(array(f_def_m)), mel2hz(array(a_filter)))           # f [Hz]
            #plt.plot(f_def_m, a_filter)                                        # f [mel]
            
        plt.grid()
        plt.xlabel('frequency [Hz]')
        plt.ylabel('amplitude')
        #plt.ylim([0, 1])
        plt.tight_layout()
        plt.show()

    return filters


if __name__ == '__main__':
    
    ''' Settings '''
    args = parse_arguments()
    input_file = args.input_file                    # .wav file to process
    ms_dur = args.microsegment_dur                  # microsegment duration [s]
    ms_shift_dur = args.shift_dur                   # microsegment shifting duration [s]
    verbose = args.verbose                          # print analysis and plot the mfcc?
    show_filters = args.show_filters                # show the mfcc filters?
    
    ''' Read, normalize and analyze wav signal '''
    f_s, s_raw = read_wav(input_file)               # sampling frequency [Hz], raw signal
    s = normalize(s=s_raw)                          # signal mapped to <-1,1>
    s_len = len(s)                                  # signal length
    s_dur = float(s_len)/f_s                        # signal duration [s]
    ms_len = int((ms_dur/s_dur) * s_len)            # microsegment length
    ms_shift = int((ms_shift_dur/s_dur) * s_len)    # microsegment shift
    n_ms = int((s_len - ms_len) / ms_shift)         # number of microsegments (windows)
    bw = f_s/2                                      # transmitted band [Hz]
    bw_m = int(hz2mel(bw))                          # transmitted band [mel]
    n_filters = 40                                  # number of filters
    db_m = float(bw_m)/(n_filters+1)                # delta m (for filter shift in mels)
    filter_len = (ms_len/2)                         # filter and also the final microsegment length
    f_def_step = float(bw)/filter_len               # frequency step, where the filter is defined [Hz]
    f_def_step_m = float(bw_m)/filter_len           # frequency step, where the filter is defined [mel]

    ''' Design filters '''
    filters = design_filters(show_filters=show_filters)
    
    ''' Loop over microsegments and get MFCC '''
    mfcc = list()
    for ms_i in range(0, n_ms+1):
        ms_i_start = ms_i * ms_shift
        ms = s[ms_i_start:ms_i_start + ms_len]       # microsegment
        ms = multiply(ms, hamming(M=ms_len))         # microsegment * hamming window
        ms = abs(fft.rfft(ms))                       # absolute value of Discrete Fourier Transform applied on microsegment*hamming

        energies = list()
        for a_filter in filters:
            energies.append(log10(dot(ms, a_filter)))
        #mfcc.append(scipy_dct(energies))
        mfcc.append(energies)

    if verbose:
        print '\n------ SIGNAL ANALYSIS ------'
        print 'microsegment duration [s] :=', ms_dur
        print 'microsegment shift duration [s] :=', ms_shift_dur
        print 'signal length:', s_len
        print 'signal duration [s]:', s_dur
        print 'sampling frequency [Hz]:', f_s
        print 'transmitted band [Hz]: 0 ÷', bw
        print 'transmitted band [mel]: 0 ÷', bw_m
        print 'microsegment (window) length:', ms_len
        print 'microsegment (window) shift:', ms_shift
        print 'number of windows:', n_ms
        print 'number of filters :=', n_filters
        print 'db [mel] :', db_m
        print 'filter (and also final microsegment) length:', filter_len
        print 'filter defined step [Hz]:', f_def_step
        print 'filter defined step [mel]:', f_def_step_m
        print '-----------------------------\n'
        print '\n ----- MEL FREQUENCE CEPSTRAL COEFFICIENTs for', input_file, '-----\n'
        print len(mfcc), mfcc[0]
        exit()
        for i, c_i in enumerate(mfcc):
            print 't =', i, ':', c_i
        
        plt.imshow(map(list, zip(*mfcc)), interpolation='nearest', aspect='auto')
        plt.title(input_file)
        plt.xlabel('t')
        plt.xlim([0, len(mfcc)])
        plt.ylabel('cepstral coefficient')
        plt.ylim([1, len(mfcc[0])-1])
        plt.yticks(range(len(mfcc[0])), range(1, len(mfcc[0])+1))
        plt.colorbar()
        plt.savefig(input_file.rstrip('.wav')+'.png')
        plt.show()