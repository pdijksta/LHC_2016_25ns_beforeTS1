
dict_warmup_cells = {}

dict_warmup_cells['S23'] = '13R2_947 31R2_947 33R2_947'.split()
dict_warmup_cells['S34'] = ['13R3_947']
dict_warmup_cells['S45'] = '13L5_947 15L5_947'.split()
dict_warmup_cells['S56'] = '13R5_947 13L6_947 33R5_947'.split()
dict_warmup_cells['S67'] = '13R6_947 17L7_947'.split()
dict_warmup_cells['S78'] = '13R7_947 13L8_947'.split()
dict_warmup_cells['S81'] = '13R8_947 31R8_947'.split()
    
# generated with pytmber script h001_get_temp_list_using_pytimber.py (Thanks PyTimber!!!!!)
dict_temp_warmup_cells={\
'S23': [u'QRLAA_13R2_TT947.TEMPERATURECALC', u'QRLAC_31R2_TT947.TEMPERATURECALC', u'QRLAD_33R2_TT947.TEMPERATURECALC'], 
'S81': [u'QRLAA_13R8_TT947.TEMPERATURECALC', u'QRLAC_31R8_TT947.TEMPERATURECALC'], 
'S34': [u'QRLAA_13R3_TT947.TEMPERATURECALC'], 
'S45': [u'QRLAA_13L5_TT947.TEMPERATURECALC', u'QRLAB_15L5_TT947.TEMPERATURECALC'], 
'S56': [u'QRLAA_13R5_TT947.TEMPERATURECALC', u'QRLAA_13L6_TT947.TEMPERATURECALC', u'QRLAD_33R5_TT947.TEMPERATURECALC'], 
'S78': [u'QRLAA_13R7_TT947.TEMPERATURECALC', u'QRLAA_13L8_TT947.TEMPERATURECALC'], 
'S67': [u'QRLAA_13R6_TT947.TEMPERATURECALC', u'QRLAA_17L7_TT947.TEMPERATURECALC']}


def vars_temp_warmup_cells():
    varlist = []
    for kk in dict_temp_warmup_cells.keys():
        varlist+=dict_temp_warmup_cells[kk]
    return varlist
