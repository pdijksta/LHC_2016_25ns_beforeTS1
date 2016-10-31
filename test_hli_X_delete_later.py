import numpy as np
import matplotlib.pyplot as plt
import LHCMeasurementTools.heat_load_impedance as hli
import LHCMeasurementTools.TimberManager as tm

filln = 5219

fill_dict = {}
fill_dict.update(tm.parse_timber_file('./fill_basic_data_csvs/basic_data_fill_%d.csv' % filln, verbose=False))
fill_dict.update(tm.parse_timber_file('./fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv' % filln, verbose=False))


hli_ob = hli.HeatLoadImpedance()

imp_hl = hli_ob.P_RW_Wm_1beam(1.15e11, 0.25e-9, 20, 27e3, 18.4e-3, n_bunches=2808, b_field=0)

print('Imp hl is %.2f mW' % (imp_hl*1e3))


if True:
    imp_hl_fill = hli_ob.P_RW_Wm_1beam_fill_arc_half_cell(fill_dict)
    sr_hl_fill = hli.HeatLoadSynchRad(fill_dict=fill_dict).avg_arc_heat_load_m

    plt.figure()
    plt.plot(hli_ob.t_stamps, imp_hl_fill)
    plt.show()


