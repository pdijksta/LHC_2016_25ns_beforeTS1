import LHCMeasurementTools.LHC_Luminosity_Calculator as llc




# parameters at injection

energy_eV = 6500e9
betastar = 0.4
crossing_angle_full_rad = 370e-6
emittance_blowup_budjet = 1. 
loss_budjet = 1.0
filln = 4947
N_colliding_bunches = 1165
ppb = 0.94e11
bunch_length_4sigma_s = 1.2e-9
nemitt_tr = 2.6e-6

energy_eV = 6500e9
betastar = 0.4
crossing_angle_full_rad = 370e-6
emittance_blowup_budjet = 1.
loss_budjet = 1.0
filln = 4958
N_colliding_bunches = 1453
ppb = .95e11
bunch_length_4sigma_s = 1.2e-9
nemitt_tr = 2.6e-6




lumi =  llc.compute_luminosity(N_colliding_bunches, ppb*loss_budjet, bunch_length_4sigma_s, nemitt_tr*emittance_blowup_budjet, energy_eV, betastar, crossing_angle_full_rad)


print 'Fill %d theoretical lumi =%f x 10^33 Hz/cm2'%(filln, lumi/1e33)
