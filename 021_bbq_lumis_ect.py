import sys, os
BIN = os.path.expanduser("../")
sys.path.append(BIN)

import pickle
import time
import pylab as pl
import numpy as np

import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.LHC_Energy as Energy
import LHCMeasurementTools.mystyle as ms
from LHCMeasurementTools.LHC_FBCT import FBCT
from LHCMeasurementTools.LHC_BCT import BCT
from LHCMeasurementTools.LHC_BQM import filled_buckets, blength
import LHCMeasurementTools.LHC_Heatloads as HL
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables

import scipy.io as sio

flag_first_on_top = True

t_toll_eneloss = 120. #s

flag_normalize_to_bct = True

#~ b

#~ filln=4231
#~ filln=4236
#~ filln=4237
#~ filln=4243
#~ filln=4246
#~ filln=4245
#~ filln=4249
#~ filln=4251
#~ filln=4252
#~ filln=4260
#~ filln=4261


if len(sys.argv)>1:
    print '--> Processing fill {:s}'.format(sys.argv[1])
    filln = int(sys.argv[1])

myfontsz = 16




with open('fills_and_bmodes.pkl', 'rb') as fid:
        dict_fill_bmodes = pickle.load(fid)
        
dict_fill_data = {}
dict_fill_data.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=True))
dict_fill_data.update(tm.parse_timber_file('fill_extra_data_csvs/extra_data_fill_%d.csv'%filln, verbose=True))


dict_beam = dict_fill_data
dict_fbct = dict_fill_data


colstr = {}
colstr[1] = 'b'
colstr[2] = 'r'



energy = Energy.energy(dict_fill_data, beam=1)

t_fill_st = dict_fill_bmodes[filln]['t_startfill']
t_fill_end = dict_fill_bmodes[filln]['t_endfill']
t_fill_len = t_fill_end - t_fill_st


t_min = dict_fill_bmodes[filln]['t_startfill']-0*60.
t_max = dict_fill_bmodes[filln]['t_endfill']+0*60.


t_ref=t_fill_st
tref_string=time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))


pl.close('all')
ms.mystyle_arial(fontsz=myfontsz, dist_tick_lab=8)



bct_bx = {}


for beam_n in [1,2]:
        bct_bx[beam_n] = BCT(dict_fill_data, beam = beam_n)


sp1 = None

fillref = 'Fill. %d started on %s'%(filln, tref_string)


#########################################
plot_title = 'Luminosity'
#########################################
fig = pl.figure(1, figsize=(12, 10))
fig.patch.set_facecolor('w')

sptotint = pl.subplot(3,1,1, sharex=sp1)
sp1=sptotint
spene = sptotint.twinx()

spene.plot((energy.t_stamps-t_ref)/3600., energy.energy, '-k')

for beam_n in [1,2]:
	sptotint.plot((bct_bx[beam_n].t_stamps-t_ref)/3600., bct_bx[beam_n].values, '-', color=colstr[beam_n])
sptotint.set_ylabel('Total intensity [p+]')
sptotint.grid('on')

splumi = pl.subplot(3,1,2, sharex=sp1)
for kk in ['ATLAS:LUMI_TOT_INST', 'CMS:LUMI_TOT_INST', 'LHCB:LUMI_TOT_INST', 'ALICE:LUMI_TOT_INST']:
	pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values())
pl.grid('on')
pl.ylabel('Luminosity [(ub*s)^-1]')

spbetastar = pl.subplot(3,1,3, sharex=sp1)
for kk in ['HX:BETASTAR_IP1', 'HX:BETASTAR_IP5', 'HX:BETASTAR_IP8', 'HX:BETASTAR_IP2']:
	pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values())
pl.grid('on')
pl.ylabel('Beta* [cm]')
pl.xlabel('Time [h]')

pl.suptitle(plot_title+'\n'+fillref)
pl.gcf().canvas.set_window_title(plot_title)

#########################################
plot_title = 'Tune feedback tunes'
#########################################
pl.figure(2, figsize=(12, 10))

sptotint = pl.subplot(3,1,1, sharex=sp1)
spene = sptotint.twinx()

spene.plot((energy.t_stamps-t_ref)/3600., energy.energy, '-k')

for beam_n in [1,2]:
	sptotint.plot((bct_bx[beam_n].t_stamps-t_ref)/3600., bct_bx[beam_n].values, '-', color=colstr[beam_n])
sptotint.set_ylabel('Total intensity [p+]')
sptotint.grid('on')

pl.subplot(3,1,2, sharex=sp1)
kk='LHC.BOFSU:TUNE_B1_H';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.b')
kk='LHC.BOFSU:TUNE_B2_H';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.r')
pl.grid('on')
pl.subplot(3,1,3, sharex=sp1)
kk='LHC.BOFSU:TUNE_B1_V';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.b')
kk='LHC.BOFSU:TUNE_B2_V';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.r')
pl.grid('on')
pl.suptitle(plot_title+'\n'+fillref)
pl.gcf().canvas.set_window_title(plot_title)

#########################################
plot_title = 'High sensitivity tunes'
#########################################
pl.figure(3, figsize=(12, 10))

sptotint = pl.subplot(3,1,1, sharex=sp1)
spene = sptotint.twinx()

spene.plot((energy.t_stamps-t_ref)/3600., energy.energy, '-k')

for beam_n in [1,2]:
	sptotint.plot((bct_bx[beam_n].t_stamps-t_ref)/3600., bct_bx[beam_n].values, '-', color=colstr[beam_n])
sptotint.set_ylabel('Total intensity [p+]')
sptotint.grid('on')

pl.subplot(3,1,2, sharex=sp1)
kk='LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_FREQ_1';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.b')
kk='LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_FREQ_1';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.r')
pl.grid('on')
pl.subplot(3,1,3, sharex=sp1)
kk='LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_FREQ_2';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.b')
kk='LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_FREQ_2';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.r')
pl.grid('on')
pl.suptitle(plot_title+'\n'+fillref)
pl.gcf().canvas.set_window_title(plot_title)


#########################################
plot_title = 'Gated tunes'
######################################### 
pl.figure(4, figsize=(12, 10))

sptotint = pl.subplot(3,1,1, sharex=sp1)
spene = sptotint.twinx()

spene.plot((energy.t_stamps-t_ref)/3600., energy.energy, '-k')

for beam_n in [1,2]:
	sptotint.plot((bct_bx[beam_n].t_stamps-t_ref)/3600., bct_bx[beam_n].values, '-', color=colstr[beam_n])
sptotint.set_ylabel('Total intensity [p+]')
sptotint.grid('on')

pl.subplot(3,1,2, sharex=sp1)
kk='LHC.BQBBQ.CONTINUOUS.B1:EIGEN_FREQ_1';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.b')
kk='LHC.BQBBQ.CONTINUOUS.B2:EIGEN_FREQ_1';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.r')
pl.grid('on')
pl.subplot(3,1,3, sharex=sp1)
kk='LHC.BQBBQ.CONTINUOUS.B1:EIGEN_FREQ_2';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.b')
kk='LHC.BQBBQ.CONTINUOUS.B2:EIGEN_FREQ_2';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.r')
pl.grid('on')
pl.suptitle(plot_title+'\n'+fillref)
pl.gcf().canvas.set_window_title(plot_title)

#########################################
plot_title = 'High sensitivity amplitudes'
######################################### 
pl.figure(5, figsize=(12, 10))

sptotint = pl.subplot(3,1,1, sharex=sp1)
spene = sptotint.twinx()

spene.plot((energy.t_stamps-t_ref)/3600., energy.energy, '-k')

for beam_n in [1,2]:
	sptotint.plot((bct_bx[beam_n].t_stamps-t_ref)/3600., bct_bx[beam_n].values, '-', color=colstr[beam_n])
sptotint.set_ylabel('Total intensity [p+]')
sptotint.grid('on')

pl.subplot(3,1,2, sharex=sp1)
kk='LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_AMPL_1';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.b')
kk='LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_AMPL_1';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.r')
pl.grid('on')
pl.subplot(3,1,3, sharex=sp1)
kk='LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_AMPL_2';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.b')
kk='LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_AMPL_2';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.r')
pl.suptitle(plot_title+'\n'+fillref)
pl.gcf().canvas.set_window_title(plot_title)
pl.grid('on')


#########################################
plot_title = 'Continuous amplitudes'
######################################### 
pl.figure(6, figsize=(12, 10))

sptotint = pl.subplot(3,1,1, sharex=sp1)
spene = sptotint.twinx()

spene.plot((energy.t_stamps-t_ref)/3600., energy.energy, '-k')

for beam_n in [1,2]:
	sptotint.plot((bct_bx[beam_n].t_stamps-t_ref)/3600., bct_bx[beam_n].values, '-', color=colstr[beam_n])
sptotint.set_ylabel('Total intensity [p+]')
sptotint.grid('on')

pl.subplot(3,1,2, sharex=sp1)
kk='LHC.BQBBQ.CONTINUOUS.B1:EIGEN_AMPL_1';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.b')
kk='LHC.BQBBQ.CONTINUOUS.B2:EIGEN_AMPL_1';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.r')
pl.grid('on')
pl.subplot(3,1,3, sharex=sp1)
kk='LHC.BQBBQ.CONTINUOUS.B1:EIGEN_AMPL_2';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.b')
kk='LHC.BQBBQ.CONTINUOUS.B2:EIGEN_AMPL_2';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.r')
pl.grid('on')
pl.suptitle(plot_title+'\n'+fillref)
pl.gcf().canvas.set_window_title(plot_title)



#########################################
plot_title = 'Trims tune feedback'
######################################### 
pl.figure(7, figsize=(12, 10))

sptotint = pl.subplot(3,1,1, sharex=sp1)
spene = sptotint.twinx()

spene.plot((energy.t_stamps-t_ref)/3600., energy.energy, '-k')

for beam_n in [1,2]:
	sptotint.plot((bct_bx[beam_n].t_stamps-t_ref)/3600., bct_bx[beam_n].values, '-', color=colstr[beam_n])
sptotint.set_ylabel('Total intensity [p+]')
sptotint.grid('on')

pl.subplot(3,1,2, sharex=sp1)
kk='LHC.BOFSU:TUNE_TRIM_B1_H';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.b')
kk='LHC.BOFSU:TUNE_TRIM_B2_H';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.r')
pl.grid('on')
pl.subplot(3,1,3, sharex=sp1)
kk='LHC.BOFSU:TUNE_TRIM_B1_V';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.b')
kk='LHC.BOFSU:TUNE_TRIM_B2_V';pl.plot((dict_fill_data[kk].t_stamps-t_ref)/3600., dict_fill_data[kk].float_values(), '.r')
pl.grid('on')
pl.suptitle(plot_title+'\n'+fillref)
pl.gcf().canvas.set_window_title(plot_title)


pl.show()
