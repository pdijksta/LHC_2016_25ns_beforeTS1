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

filln = 5085
filln = 5038
filln = 4929
filln = 5112

slot_min = 300
slot_max = 3500

every_n_bun = 1

mode = 'injection'
#mode = 'wholefill'
#~ mode = 'first2sb'

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

if mode == 'injection':
    t_start_obs = dict_fill_bmodes[filln]['t_start_INJPHYS']
    t_end_obs = dict_fill_bmodes[filln]['t_stop_INJPHYS']
elif mode == 'wholefill':
    t_start_obs = dict_fill_bmodes[filln]['t_start_INJPHYS']
    t_end_obs = t_end_fill
elif mode == 'first2sb':
    t_start_obs = dict_fill_bmodes[filln]['t_start_STABLE']
    t_end_obs = dict_fill_bmodes[filln]['t_start_STABLE']+2*3600
else:
    raise ValueError('Mode not understood!')
    
t_obs_vect = np.array([t_start_obs, t_end_obs])

bsrt_calib_dict = BSRT_calib.emittance_dictionary()

fill_dict = {}
fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=True))
fill_dict.update(tm.parse_timber_file('fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv'%filln, verbose=True))

pl.close('all')

colstr = {}
colstr[1] = 'b'
colstr[2] = 'r'

sp1 = None

#~ for beam_n in [1]:
beam_n = 1
pl.figure(beam_n)

energy = Energy.energy(fill_dict, beam=beam_n)
bct = BCT.BCT(fill_dict, beam=beam_n)
bsrt  = BSRT.BSRT(fill_dict, beam=beam_n, calib_dict=bsrt_calib_dict, average_repeated_meas=True)

bsrt.calculate_emittances(energy)

dict_bunches, t_bbb, emit_h_bbb, emit_v_bbb, bunch_n_un = bsrt.get_bbb_emit_evolution()

sptotint = pl.subplot(3,1,1, sharex=sp1)
sp1 = sptotint
spenergy = sptotint.twinx()

spenergy.plot((energy.t_stamps-t_ref), energy.energy/1e3, c='black', lw=2.)#, alpha=0.1)
spenergy.set_ylabel('Energy [TeV]')
spenergy.set_ylim(0,7)

sptotint.plot((bct.t_stamps-t_ref), bct.values, '-', color=colstr[beam_n], lw=2.)
sptotint.set_ylabel('Total intensity [p+]')
sptotint.grid('on')

spemit_h = pl.subplot(3,1,2, sharex=sp1)
spemit_v = pl.subplot(3,1,3, sharex=sp1)

mask_obs = np.logical_and(bunch_n_un>slot_min, bunch_n_un<slot_max)
slots_obs = bunch_n_un[mask_obs][::every_n_bun]

pl.figure(100)
sp_inj_time = pl.subplot(3,1,1)
sp_inj_emith = pl.subplot(3,1,2, sharex = sp_inj_time)
sp_inj_emitv = pl.subplot(3,1,3, sharex = sp_inj_time)

pl.figure(101)
sp_inj_time2 = pl.subplot(2,1,1, sharex= sp_inj_time)
sp_inj_diff = pl.subplot(2,1,2, sharex = sp_inj_time)

#~ pl.figure(200)
#~ sp_buinj = pl.subplot(1,1,1)

mh_list = []
mv_list = []

bh_list = []
bv_list = []

for i_trace, slot_bun in enumerate(slots_obs):

    mask_obs = np.logical_and(dict_bunches[slot_bun]['t_stamp']>t_start_obs, dict_bunches[slot_bun]['t_stamp']<t_end_obs)
    
    if sum(mask_obs)>0:
        
        t_stamp_curr = dict_bunches[slot_bun]['t_stamp'][mask_obs]
        norm_emit_h_curr = dict_bunches[slot_bun]['norm_emit_h'][mask_obs]
        norm_emit_v_curr = dict_bunches[slot_bun]['norm_emit_v'][mask_obs]
        
        colorcurr = ms.colorprog(i_trace, len(slots_obs))
        spemit_h.plot((t_stamp_curr-t_ref), norm_emit_h_curr, '.-', color = colorcurr)
        spemit_v.plot((t_stamp_curr-t_ref), norm_emit_v_curr, '.-', color = colorcurr)
        
        mh, bh = np.polyfit(t_stamp_curr, norm_emit_h_curr, 1)
        mv, bv = np.polyfit(t_stamp_curr, norm_emit_v_curr, 1)
        
        #~ spemit_h.plot((t_obs_vect-t_ref)/3600., mh*t_obs_vect+bh, '-', color = colorcurr)
        #~ spemit_v.plot((t_obs_vect-t_ref)/3600., mv*t_obs_vect+bv, '-', color = colorcurr)
        
        mh_list.append(mh)
        mv_list.append(mv)

        bh_list.append(bh)
        bv_list.append(bv)
        
        sp_inj_emith.plot(slot_bun, norm_emit_h_curr[0], '.k')
        sp_inj_emith.plot(slot_bun, norm_emit_h_curr[-1], '.g')
        
        sp_inj_emitv.plot(slot_bun, norm_emit_v_curr[0], '.k')
        sp_inj_emitv.plot(slot_bun, norm_emit_v_curr[-1], '.g')
        
        
        sp_inj_time.plot(slot_bun,(t_stamp_curr[-1]-t_stamp_curr[0])/60., '.g')
        
        #~ sp_buinj.plot((t_stamp_curr[-1]-t_stamp_curr[0])/60., (norm_emit_h_curr[-1]-norm_emit_h_curr[0]), '.b', markersize=.6)
        #~ sp_buinj.plot((t_stamp_curr[-1]-t_stamp_curr[0])/60., (norm_emit_v_curr[-1]-norm_emit_v_curr[0]), '.r', markersize=.6)
        
        sp_inj_time2.plot(slot_bun,(t_stamp_curr[-1]-t_stamp_curr[0])/60., '.g')
        sp_inj_diff.plot(slot_bun, (norm_emit_h_curr[-1]-norm_emit_h_curr[0]), '.b')
        sp_inj_diff.plot(slot_bun, (norm_emit_v_curr[-1]-norm_emit_v_curr[0]), '.r')
    else:
        mh_list.append(0.)
        mv_list.append(0.)

        bh_list.append(0.)
        bv_list.append(0.)


sp_inj_diff.grid('on')

spemit_h.set_ylim(1.5, 4.)
spemit_v.set_ylim(1.5, 4.)

pl.figure(2)
spslope_h = pl.subplot(2,1,1)
spslope_v = pl.subplot(2,1,2, sharex=spslope_h)

spslope_h.plot(slots_obs, np.array(mh_list)*3600, '.')
spslope_v.plot(slots_obs, np.array(mv_list)*3600, '.')

hist_h, bin_edges_h = np.histogram(np.array(mh_list)*3600, range =(-5,5), bins=120)
hist_v, bin_edges_v = np.histogram(np.array(mv_list)*3600, range =(-5,5), bins=120)
pl.figure(3)
sphisth = pl.subplot(2,1,1)
pl.bar(bin_edges_h[:-1], hist_h, width=np.diff(bin_edges_h))
pl.subplot(2,1,2, sharex=sphisth)
pl.bar(bin_edges_v[:-1], hist_v, width=np.diff(bin_edges_v))
pl.show()
