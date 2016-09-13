import LHCMeasurementTools.LHC_BCT as BCT
import LHCMeasurementTools.LHC_Energy as Energy
import LHCMeasurementTools.LHC_BSRT as BSRT
import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.mystyle as ms
import BSRT_calib 
import numpy as np
import pylab as pl
import pickle
import sys, time
from colorsys import hsv_to_rgb

# BSRT scan parameters
filln = 4914
list_scan_times = np.linspace(1.7, 2.25, 20)

scan_thrld = 100
plot_emittance = True

average_repeated_meas = False


fills_bmodes_file = 'fills_and_bmodes.pkl'
with open(fills_bmodes_file, 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)


if len(sys.argv)>1:
     print '--> Processing fill {:s}'.format(sys.argv[1])
     filln = int(sys.argv[1])

t_start_fill = dict_fill_bmodes[filln]['t_startfill']
t_end_fill = dict_fill_bmodes[filln]['t_endfill']
t_fill_len = t_end_fill - t_start_fill
t_ref = t_start_fill
n_traces = 50.

bsrt_calib_dict = BSRT_calib.emittance_dictionary()

if len(sys.argv)>1:

     if np.any(map(lambda s: ('--n_traces'in s), sys.argv)):
        i_arg = np.where(map(lambda s: ('--n_traces'in s), sys.argv))[0]
        arg_temp = sys.argv[i_arg]
        n_traces = float(arg_temp.split('=')[-1])

        list_scan_times = np.linspace((t_start_fill-t_ref)/3600., (t_end_fill-t_ref)/3600., n_traces)
         
     if '--injection' in sys.argv:
                print 'Scans in the INJPHYS-PRERAMP beam modes'
                t_start_INJPHYS = dict_fill_bmodes[filln]['t_start_INJPHYS']
                t_start_RAMP = dict_fill_bmodes[filln]['t_start_RAMP']
                list_scan_times = np.linspace((t_start_INJPHYS-t_ref)/3600., (t_start_RAMP-t_ref)/3600., n_traces)


     if '--highenergy' in sys.argv:
                print 'Scans in the FLATTOP-STABLE beam modes'
                t_start_FLATTOP = dict_fill_bmodes[filln]['t_start_FLATTOP']
                t_start_STABLE = dict_fill_bmodes[filln]['t_start_STABLE']
                list_scan_times = np.linspace((t_start_FLATTOP-t_ref)/3600., (t_start_STABLE-t_ref)/3600.+0.5, n_traces)

     if '--stablebeams' in sys.argv:
                print 'Scans in the STABLE BEAMS'
                t_start_STABLE = dict_fill_bmodes[filln]['t_start_STABLE']
                t_end_STABLE = dict_fill_bmodes[filln]['t_stop_STABLE']
                list_scan_times = np.linspace((t_start_STABLE-t_ref)/3600., (t_end_STABLE-t_ref)/3600.+0.5, n_traces)

     if '--sigma' in sys.argv:
                plot_emittance=False
                
     if '--avgrep' in sys.argv:
            average_repeated_meas = True

     if np.any(map(lambda s: ('--interval'in s), sys.argv)):
        i_arg = np.where(map(lambda s: ('--interval'in s), sys.argv))[0]
        arg_temp = sys.argv[i_arg]
        t_start_man = float(arg_temp.split('=')[-1].split(',')[0])
        t_end_man = float(arg_temp.split('=')[-1].split(',')[1])
        print 'Interval manually set: %.2fh to %.2fh'%(t_start_man, t_end_man)
        list_scan_times = np.linspace(t_start_man, t_end_man, n_traces)

     if '--notrace' in sys.argv:
        list_scan_times = []


fill_dict = {}
fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=True))
fill_dict.update(tm.parse_timber_file('fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv'%filln, verbose=True))

beam_col = ['k', 'b','r']

sp_sigma_h = None
pl.close('all')
for beam in [1,2]:
    energy = Energy.energy(fill_dict, beam=beam)
    bct = BCT.BCT(fill_dict, beam=beam)
    bsrt  = BSRT.BSRT(fill_dict, beam=beam, calib_dict=bsrt_calib_dict, average_repeated_meas=average_repeated_meas)
    if plot_emittance:
        bsrt.calculate_emittances(energy)

    # START PLOT
    fig_h = pl.figure(beam, figsize=(17,10))
    fig_h.patch.set_facecolor('w')
    ms.mystyle()

    # Intensity and energy
    sp_int = pl.subplot2grid((2,3), (0, 0), rowspan=1)
    sp_energy = sp_int.twinx()
    sp_int.grid('on')
    #~ sp_int.set_ylim(0., None)
    
    #mask_bct = bct.values > 1e12
    #~ t_start_bct = bct.t_stamps[np.min(np.where(mask_bct))]
    sp_int.plot((bct.t_stamps - t_ref)/3600., bct.values, linewidth=2, color=beam_col[beam])
    mask_ene = energy.t_stamps > t_ref
    sp_energy.plot((energy.t_stamps[mask_ene] - t_ref)/3600., energy.energy[mask_ene]/1e3, 'k', linewidth=2)
    

    sp_int.set_ylabel('Intensity [p$^+$]')
    sp_energy.set_ylabel('Energy [TeV]')
    sp_int.grid('on')
    sp_int.set_ylim(0., None)


    # Bunches
    fill = BSRT.Masked(bsrt, t_start_fill, t_end_fill)
    sp_bunch = pl.subplot2grid((2,3), (1, 0), rowspan=1, sharex = sp_int)

    mask_scan = fill.t_stamps > t_ref
    sp_bunch.plot((fill.t_stamps[mask_scan] - t_ref)/3600., fill.bunch_n[mask_scan], 'b')
    bsrt.find_start_scans(scan_thrld)
    #for t_scan in bsrt.t_start_scans:
    #    sp_bunch.axvline((float(t_scan) - t_ref)/3600., color='k')

    pl.ylabel('Acq. bunch')
    pl.xlabel('Time [h]')
    # pl.ylim(0,1200)
    # pl.xlim(0, 5.5)
    # pl.xlim(0, t_fill_len/3600.)

    # Sigma and emittance
    N_scans = len(list_scan_times)
    sp_sigma_h = pl.subplot2grid((2,3), (0, 1), rowspan=1, colspan=2, sharex = sp_sigma_h)
    sp_sigma_v = pl.subplot2grid((2,3), (1, 1), rowspan=1, colspan=2, sharex = sp_sigma_h)

    for ii in xrange(N_scans):
        colorcurr = hsv_to_rgb(float(ii)/float(N_scans), 0.9, 1.)
        colorcurr = [pl.cm.rainbow(k) for k in np.linspace(0, 1, N_scans)][ii]

        t_start_requested = list_scan_times[ii]*3600. + t_ref
        try:
                        scan = bsrt.find_closest_scan(t_start_requested, scan_thrld)
        except IndexError as err:
                        print 'Stop plotting! Got:'
                        print err
                        continue

        if plot_emittance:
            sp_sigma_h.plot(scan.bunch_n, scan.norm_emit_h, '.', color=colorcurr)
            sp_sigma_v.plot(scan.bunch_n, scan.norm_emit_v, '.', color=colorcurr)

        else:
            sp_sigma_h.plot(scan.bunch_n, scan.sigma_h, '.', color=colorcurr)
            sp_sigma_v.plot(scan.bunch_n, scan.sigma_v, '.', color=colorcurr)

        sp_bunch.axvspan((scan.t_start - t_ref)/3600., (scan.t_stop - t_ref)/3600., facecolor=colorcurr, alpha=0.8, linewidth=0)
        sp_int.axvspan((scan.t_start - t_ref)/3600., (scan.t_stop - t_ref)/3600., facecolor=colorcurr, alpha=0.8, linewidth=0)

        #sp_bunch.axvline((scan.t_start - t_ref)/3600., color=colorcurr, alpha=0.99)
        #sp_int.axvline((scan.t_start - t_ref)/3600., color=colorcurr, alpha=0.99)

    sp_sigma_h.set_xlim(0, 3300)
    sp_sigma_v.set_xlim(0, 3300)
    #sp_sigma_h.set_ylim(0, 10)
    #sp_sigma_v.set_ylim(0, 10)
    sp_sigma_h.set_xlabel('25 ns slot')
    sp_sigma_v.set_xlabel('25 ns slot')
    sp_sigma_h.grid('on')
    sp_sigma_v.grid('on')
    if plot_emittance:
        sp_sigma_h.set_ylabel('Hor. emittance [um]')
        sp_sigma_v.set_ylabel('Vert. emittance [um]')
        plot_str = 'emittance'
    else:
        sp_sigma_h.set_ylabel('Hor. sigma [a.u.]')
        sp_sigma_v.set_ylabel('Vert. sigma [a.u.]')
        plot_str = 'sigma'

    tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))
    fig_h.suptitle('Fill %d: B%d, started on %s'%(filln, bsrt.beam, tref_string), fontsize=18)
    fig_h.subplots_adjust(top=0.9,right=0.95, left=0.07, hspace=0.3, wspace=0.45)


pl.show()
