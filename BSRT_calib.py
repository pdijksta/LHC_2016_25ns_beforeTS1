
def emittance_dictionary(filln=None):

    e_dict = {'betaf_h':{}, 'betaf_v':{}, 'gamma':{}, 
          'sigma_corr_h':{}, 'sigma_corr_v':{},
          'rescale_sigma_h':{}, 'rescale_sigma_v':{}}
          
    if filln is None:
        raise ValueError('A fill number must be provided to select calibration!')
    
    if filln<5256:

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
        e_dict['rescale_sigma_h'][450][1] = 1.
        e_dict['rescale_sigma_h'][6500][1] = 1.
        e_dict['rescale_sigma_v'][450][1] = 1.
        e_dict['rescale_sigma_v'][6500][1] = 1.

        # Beam 2:
        e_dict['betaf_h'][450][2] = 200.6
        e_dict['betaf_h'][6500][2] = 200.
        e_dict['betaf_v'][450][2] = 327.1
        e_dict['betaf_v'][6500][2] = 330.
        e_dict['sigma_corr_h'][450][2] = 0.518
        e_dict['sigma_corr_h'][6500][2] = 0.299
        e_dict['sigma_corr_v'][450][2] = 0.675
        e_dict['sigma_corr_v'][6500][2] = 0.299
        e_dict['rescale_sigma_h'][450][2] = 1.
        e_dict['rescale_sigma_h'][6500][2] = 1.
        e_dict['rescale_sigma_v'][450][2] = 1.
        e_dict['rescale_sigma_v'][6500][2] = 1.

        # gamma
        e_dict['gamma'][450] = 479.6 
        e_dict['gamma'][6500] = 6927.6
        
        print 'Using calibration A'


        
    elif filln>=5256 and filln<5405:

        for kk in e_dict.keys():
            e_dict[kk] = {450:{}, 6500:{}}

        # Beam 1:
        e_dict['betaf_h'][450][1] = 204.1
        e_dict['betaf_h'][6500][1] = 200.
        e_dict['betaf_v'][450][1] = 317.3
        e_dict['betaf_v'][6500][1] = 330.
        e_dict['sigma_corr_h'][450][1] = .53
        e_dict['sigma_corr_h'][6500][1] = .31
        e_dict['sigma_corr_v'][450][1] = .59
        e_dict['sigma_corr_v'][6500][1] = .31
        e_dict['rescale_sigma_h'][450][1] = .977
        e_dict['rescale_sigma_h'][6500][1] = 1.0232
        e_dict['rescale_sigma_v'][450][1] = .94
        e_dict['rescale_sigma_v'][6500][1] = .9375

        # Beam 2:
        e_dict['betaf_h'][450][2] = 200.6
        e_dict['betaf_h'][6500][2] = 200.
        e_dict['betaf_v'][450][2] = 327.1
        e_dict['betaf_v'][6500][2] = 330.
        e_dict['sigma_corr_h'][450][2] = .48
        e_dict['sigma_corr_h'][6500][2] = .31
        e_dict['sigma_corr_v'][450][2] = .48
        e_dict['sigma_corr_v'][6500][2] = .26
        e_dict['rescale_sigma_h'][450][2] = 1.0192
        e_dict['rescale_sigma_h'][6500][2] = 1.0204
        e_dict['rescale_sigma_v'][450][2] = .9655
        e_dict['rescale_sigma_v'][6500][2] = .9821

        # gamma
        e_dict['gamma'][450] = 479.6 
        e_dict['gamma'][6500] = 6927.6
        print 'Using calibration B'
        
    elif filln>=5405: 
        for kk in e_dict.keys():
            e_dict[kk] = {450:{}, 6500:{}}

        # Beam 1:
        e_dict['betaf_h'][450][1] = 204.1
        e_dict['betaf_h'][6500][1] = 200.
        e_dict['betaf_v'][450][1] = 317.3
        e_dict['betaf_v'][6500][1] = 330.
        e_dict['sigma_corr_h'][450][1] = .53
        e_dict['sigma_corr_h'][6500][1] = .31
        e_dict['sigma_corr_v'][450][1] = .59
        e_dict['sigma_corr_v'][6500][1] = .31
        e_dict['rescale_sigma_h'][450][1] = 1.
        e_dict['rescale_sigma_h'][6500][1] = 1.
        e_dict['rescale_sigma_v'][450][1] = 1.
        e_dict['rescale_sigma_v'][6500][1] = 1.

        # Beam 2:
        e_dict['betaf_h'][450][2] = 200.6
        e_dict['betaf_h'][6500][2] = 200.
        e_dict['betaf_v'][450][2] = 327.1
        e_dict['betaf_v'][6500][2] = 330.
        e_dict['sigma_corr_h'][450][2] = .48
        e_dict['sigma_corr_h'][6500][2] = .31
        e_dict['sigma_corr_v'][450][2] = .48
        e_dict['sigma_corr_v'][6500][2] = .26
        e_dict['rescale_sigma_h'][450][2] = 1.
        e_dict['rescale_sigma_h'][6500][2] = 1.
        e_dict['rescale_sigma_v'][450][2] = 1.
        e_dict['rescale_sigma_v'][6500][2] = 1.

        # gamma
        e_dict['gamma'][450] = 479.6 
        e_dict['gamma'][6500] = 6927.6 
        print 'Using calibration C'
    else:         
        raise ValueError('What?!')     
    
         
    
    return(e_dict)
