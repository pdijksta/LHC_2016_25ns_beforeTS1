
def emittance_dictionary():
    e_dict = {'betaf_h':{}, 'betaf_v':{}, 'gamma':{}, 
              'sigma_corr_h':{}, 'sigma_corr_v':{}}
    for kk in e_dict.keys():
        e_dict[kk] = {450:{}, 6500:{}}

    # Beam 1:
    e_dict['betaf_h'][450][1] = 204.1
    e_dict['betaf_h'][6500][1] = 200.
    e_dict['betaf_v'][450][1] = 317.3
    e_dict['betaf_v'][6500][1] = 330.
    e_dict['sigma_corr_h'][450][1] = 0.528
    e_dict['sigma_corr_h'][6500][1] = 0.303 
    e_dict['sigma_corr_v'][450][1] = 0.437
    e_dict['sigma_corr_v'][6500][1] = 0.294

    # Beam 2:
    e_dict['betaf_h'][450][2] = 200.6
    e_dict['betaf_h'][6500][2] = 200.
    e_dict['betaf_v'][450][2] = 327.1
    e_dict['betaf_v'][6500][2] = 330.
    e_dict['sigma_corr_h'][450][2] = 0.518
    e_dict['sigma_corr_h'][6500][2] = 0.299
    e_dict['sigma_corr_v'][450][2] = 0.675
    e_dict['sigma_corr_v'][6500][2] = 0.299

    # gamma
    e_dict['gamma'][450] = 479.6 
    e_dict['gamma'][6500] = 6927.6
    
    return(e_dict)
