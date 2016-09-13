import time, sys
import pylab as pl
import matplotlib.dates as dates
from scipy.integrate import cumtrapz
import numpy as np
import LHCMeasurementTools.mystyle as ms
import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.TimestampHelpers as TH
import LHCMeasurementTools.LHC_BCT as BCT
import LHCMeasurementTools.LHC_Heatloads as HL
import LHCMeasurementTools.LHC_Energy as Energy
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables
from LHCMeasurementTools.LHC_Fills import Fills_Info


empty_fills = [4901, 4904, 4969]
device_blacklist = [\
'QRLAA_33L5_QBS947_D4.POSST']



# Specify plot settings
t_start_plot = '31-07-2015,08:00' ##
t_end_plot = '02-08-2015,08:00' ##

time_in = 'd'  #'h','d','hourtime','datetime' #
t_plot_tick_h = 12.

name_varlists_to_combine = ['AVG_ARC'] ##

normalization_to_length_of = 'magnet' #None,'magnet','cryostat' ##
mode = 'norm_to_intensity' #'norm_to_intensity','integrated' ##

plot_all = True ##
plot_average = True ##
plot_model = True ##
flag_filln = True ##

zero_at = None

screen_mode = 'CCC' #'small','CCC' ##


int_cut_norm = 1e13
dictYN = {'Y':True, 'N':False, 'y':True, 'n':False}

print sys.argv
param = 'tstart'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg]
    t_start_plot = arg_temp.split('=')[-1]
        
param = 'tend'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg]
    t_end_plot = arg_temp.split('=')[-1]

param = 'tagfills'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    flag_filln = dictYN[arg_temp]
    
param = 'plotall'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    plot_all = dictYN[arg_temp]
    
param = 'plotaverage'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    plot_average = dictYN[arg_temp]

param = 'varlists'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    name_varlists_to_combine = arg_temp.split(',')
    
param = 'plotmodel'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    plot_model = dictYN[arg_temp]

param = 'screen'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    screen_mode = arg_temp

param = 'mode'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    mode = arg_temp 
    
param = 'normlength'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    normalization_to_length_of = arg_temp  
    if 'None' in normalization_to_length_of or 'none' in normalization_to_length_of:
        normalization_to_length_of = None
    
param =  'timein' 
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    time_in = arg_temp

param =  'hourtickspac'     
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    t_plot_tick_h = float(arg_temp)


if screen_mode == 'small':
    fontsz = 14
    fontsz_leg = 14
    figsz = (12,9)
elif screen_mode == 'CCC':
    fontsz = 16
    fontsz_leg = 16
    figsz = (15,9*5/4.)
else:
    raise ValueError('Screen mode not recognized!')


hl_varlist = []
for name_varlist in name_varlists_to_combine:
    hl_varlist += HL.variable_lists_heatloads[name_varlist]

for varname in hl_varlist:
    if varname in device_blacklist:
        hl_varlist.remove(varname)

param = 'plotmodel'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    plot_model = dictYN[arg_temp]

param = 'screen'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    screen_mode = arg_temp

param = 'mode'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    mode = arg_temp 
    
param = 'normlength'
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    normalization_to_length_of = arg_temp  
    if 'None' in normalization_to_length_of or 'none' in normalization_to_length_of:
        normalization_to_length_of = None
    
param =  'timein' 
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    time_in = arg_temp

param =  'hourtickspac'     
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    if arg_temp=='week':
        t_plot_tick_h = arg_temp
    else:
        t_plot_tick_h = float(arg_temp)

param =  'zeroat'     
if np.any(map(lambda s: (param in s), sys.argv)):
    i_arg = np.where(map(lambda s: (param in s), sys.argv))[0][0]
    arg_temp = sys.argv[i_arg].split('=')[-1]
    zero_at = arg_temp
    
if screen_mode == 'small':
    fontsz = 14
    fontsz_leg = 14
    figsz = (12,9)
elif screen_mode == 'CCC':
    fontsz = 15
    fontsz_leg = 15
    figsz = (15,9*5/4.)
else:
    raise ValueError('Screen mode not recognized!')


hl_varlist = []
for name_varlist in name_varlists_to_combine:
    hl_varlist += HL.variable_lists_heatloads[name_varlist]

for varname in hl_varlist:
    if varname in device_blacklist:
        hl_varlist.remove(varname)

# get magnet lengths for normalization_to_length_of
if normalization_to_length_of == 'magnet':
    norm_length_dict = HL.get_dict_magnet_lengths()
if normalization_to_length_of == 'cryostat':
    norm_length_dict = HL.get_dict_cryostat_lengths()

t_start_unix =  time.mktime(time.strptime(t_start_plot, '%d-%m-%Y,%H:%M'))
t_end_unix =  time.mktime(time.strptime(t_end_plot, '%d-%m-%Y,%H:%M'))
t_ref_unix = t_start_unix
tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref_unix))

time_conv = TH.TimeConverter(time_in, t_ref_unix, t_plot_tick_h=t_plot_tick_h)
tc = time_conv.from_unix

fill_info = Fills_Info('fills_and_bmodes.pkl')
fill_list = fill_info.fills_in_time_window(t_start_unix, t_end_unix)

# find offset to remove
if zero_at is not None:
    print 'Evaluating offsets'
    if ':' in zero_at:
        t_zero_unix = time.mktime(time.strptime(zero_at, '%d-%m-%Y,%H:%M'))
    else:
        t_zero_unix  = t_ref_unix + float(zero_at)*3600.
    filln_offset = fill_info.filln_at_time(t_zero_unix)
    
    try:
        fill_dict = tm.timber_variables_from_h5('heatloads_fill_h5s/heatloads_all_fill_%d.h5'%filln_offset)
        print 'From h5!'
    except IOError:
        print "h5 file not found, using csvs"
        fill_dict = {}
        fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln_offset, verbose=False))
        fill_dict.update(tm.parse_timber_file('fill_heatload_data_csvs/heatloads_fill_%d.csv'%filln_offset, verbose=False))

    dict_offsets={}
    for kk in hl_varlist:
        dict_offsets[kk] = np.interp(t_zero_unix, np.float_(np.array(fill_dict[kk].t_stamps)), fill_dict[kk].float_values())


pl.close('all')
ms.mystyle_arial(fontsz=fontsz, dist_tick_lab=9)
fig = pl.figure(1, figsize=figsz)
fig.patch.set_facecolor('w')
ax1 = fig.add_subplot(311)
ax11 = ax1.twinx()
ax2 = fig.add_subplot(312, sharex=ax1)
ax3 = fig.add_subplot(313, sharex=ax1)
ms.sciy()

N_fills = len(fill_list)

t_for_integrated = []
hl_for_integrated = []

first_fill = True

for i_fill, filln in enumerate(fill_list):

    print 'Fill %d, %d/%d'%(filln, i_fill+1, N_fills)
    if filln in empty_fills:
        print 'Fill blacklisted!'
        continue
    
    t_startfill = fill_info.dict_fill_bmodes[filln]['t_startfill']
    t_endfill = fill_info.dict_fill_bmodes[filln]['t_endfill']
    try:
        fill_dict = tm.timber_variables_from_h5('heatloads_fill_h5s/heatloads_all_fill_%d.h5'%filln)
        print 'From h5!'
    except IOError:
        print "h5 file not found, using csvs"
        fill_dict = {}
        fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=False))
        fill_dict.update(tm.parse_timber_file('fill_heatload_data_csvs/heatloads_fill_%d.csv'%filln, verbose=False))


    bct_b1 = BCT.BCT(fill_dict, beam=1)
    bct_b2 = BCT.BCT(fill_dict, beam=2)    
    energy = Energy.energy(fill_dict, beam=1, t_start_fill=t_startfill, t_end_fill=t_endfill)
    ax1.plot(tc(bct_b1.t_stamps), bct_b1.values*1e-14, lw=2, c='b')
    ax1.plot(tc(bct_b2.t_stamps), bct_b2.values*1e-14, lw=2, c='r')
    ax11.plot(tc(energy.t_stamps), energy.energy/1e3, c='black', lw=1.5, alpha=0.5)
    
    heatloads = SetOfHomogeneousNumericVariables(variable_list=hl_varlist, timber_variables=fill_dict)
    hl_model = SetOfHomogeneousNumericVariables(variable_list=HL.variable_lists_heatloads['MODEL'], timber_variables=fill_dict)
        
    
    # remove offset
    if zero_at is not None:
        for device in hl_varlist:
            heatloads.timber_variables[device].values = heatloads.timber_variables[device].values - dict_offsets[device]
    
    # normalize to the length
    if normalization_to_length_of is not None:
        for device in hl_varlist:
            heatloads.timber_variables[device].values = heatloads.timber_variables[device].values/norm_length_dict[device]
        for device in HL.variable_lists_heatloads['MODEL']:
            hl_model.timber_variables[device].values = hl_model.timber_variables[device].values/53.45
            

    if plot_all:
        for ii, kk in enumerate(heatloads.variable_list):
            colorcurr = ms.colorprog(i_prog=ii, Nplots=len(heatloads.variable_list))        
            if first_fill: 
                label = ''
                for st in kk.split('.POSST')[0].split('_'):
                    if 'QRL' in st or 'QBS' in st or 'AVG' in st or 'ARC' in st:
                        pass
                    else:
                        label += st + ' '
                label = label[:-1]
            else:
                label = None        
            ax2.plot(tc(heatloads.timber_variables[kk].t_stamps), heatloads.timber_variables[kk].values,
                       '-', color=colorcurr, lw=2., label=label)
            
            if mode == 'norm_to_intensity':
                t_curr = heatloads.timber_variables[kk].t_stamps
                hl_curr = heatloads.timber_variables[kk].values
                bct1_int = np.interp(t_curr, bct_b1.t_stamps, bct_b1.values)
                bct2_int = np.interp(t_curr, bct_b2.t_stamps, bct_b2.values)
                hl_norm = hl_curr/(bct1_int+bct2_int)
                hl_norm[(bct1_int+bct2_int)<int_cut_norm] = 0.
                ax3.plot(tc(t_curr), hl_norm,'-', color=colorcurr, lw=2.)
                
    if plot_model and 'AVG_ARC' in name_varlists_to_combine:
        kk = 'LHC.QBS_CALCULATED_ARC.TOTAL'
        if first_fill: label='Imp.+SR'
        else:label=None
        ax2.plot(tc(hl_model.timber_variables[kk].t_stamps), hl_model.timber_variables[kk].values,
            '--', color='grey', lw=2., label=label)

    if plot_average or mode == 'integrated':
        hl_ts, hl_aver = heatloads.mean()
    
    if plot_average:  
        ax2.plot(tc(hl_ts), hl_aver,'k-', lw=2.)
    
    if mode == 'integrated':
        t_for_integrated += list(hl_ts)
        hl_for_integrated += list(hl_aver)

    if flag_filln and t_startfill>t_start_unix-15*60:
        # Fill number labeling
        fds = tc(t_startfill)
        trans = ax1.get_xaxis_transform()
        ax1.axvline(fds, c='grey', ls='dashed', lw=2, alpha=0.4)
        ax2.axvline(fds, c='grey', ls='dashed', lw=2, alpha=0.4)
        ax3.axvline(fds, c='grey', ls='dashed', lw=2, alpha=0.4)
        try:x_fn = fds[0] #in case we are using the date
        except IndexError: x_fn = fds
        ax1.annotate('%d'%filln, xy=(x_fn, 1.01), xycoords=trans,
                            horizontalalignment='left', verticalalignment='bottom',
                            rotation=67.5, color='grey', alpha=0.8)
        
        
    first_fill = False


ax1.set_xlim(tc(t_start_unix), tc(t_end_unix))
ax11.set_ylim(0, 7)
ax1.set_ylim(0, None)
ax1.grid('on')
ax1.set_ylabel('Total intensity [10$^{14}$ p$^+$]')
ax11.set_ylabel('Energy [TeV]')
time_conv.set_x_for_plot(fig, ax1)

if normalization_to_length_of is None:
    ax2.set_ylabel('Heat load [W]')
else:
    ax2.set_ylabel('Heat load [W/m]')
ax2.legend(bbox_to_anchor=(1.06, 1.05),  loc='upper left', prop={'size':fontsz_leg})#, frameon=False)
ax2.set_ylim(0, None)
ax2.grid('on')

if mode == 'integrated':
    hl_for_integrated = np.array(hl_for_integrated)
    hl_for_integrated[hl_for_integrated<0.] = 0.
    t_for_integrated = np.array(t_for_integrated)
    hl_for_integrated[t_for_integrated < t_start_unix] = 0.
    integrated_hl = cumtrapz(hl_for_integrated, t_for_integrated)
    ax3.plot(tc(t_for_integrated[:-1]), integrated_hl,'b-', lw=2.)
    if normalization_to_length_of is None:
        ax3.set_ylabel('Integrated heat load [J]')
    else:
        ax3.set_ylabel('Integrated heat load [J/m]')
elif mode == 'norm_to_intensity':
    if normalization_to_length_of is None:
        ax3.set_ylabel('Heat load [W/p+]')
    else:
        ax3.set_ylabel('Heat load [W/m/p$^+$]')
ax3.grid('on')
ax3.set_ylim(0, None)

if time_in == 'd' or time_in == 'h':
    ax3.set_xlabel('Time [%s]'%time_in)

pl.suptitle('From ' + tref_string)
fig.subplots_adjust(left=.08, right=.82, hspace=.28, top=.89)
pl.savefig('plot.png', dpi=200)
pl.show()





