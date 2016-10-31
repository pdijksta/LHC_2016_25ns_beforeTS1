#. ../LHC_2016_25ns_followup_pytimber/start_riccardos_anaconda
import pytimber
db=pytimber.LoggingDB(clientid='OP_DATA_MINING',appid='OPTICS_ANALYSIS',source='ldb')

import warmup_cells_lists as wcl

dict_temp_warmup_cells = {}

for  sectorname in wcl.dict_warmup_cells.keys():
    currlist  = []
    for cellname in wcl.dict_warmup_cells[sectorname]:
        varname_list = db.search('%'+cellname.split('_')[0]+'%'+cellname.split('_')[-1]+'%TEMP%CALC')

        if len(varname_list)==0 or len(varname_list)>1:
            raise ValueError('There seems to be an inconsistency')
        varname = varname_list[0]
        print cellname, varname
        currlist.append(varname)
    dict_temp_warmup_cells[sectorname] = currlist
    
    
dict_hl_warmup_cells = {}
for  sectorname in wcl.dict_warmup_cells.keys():
    currlist  = []
    for cellname in wcl.dict_warmup_cells[sectorname]:
        varname_list = db.search('%'+cellname.split('_')[0]+'%QBS'+cellname.split('_')[-1]+'%POSST')

        if len(varname_list)==0 or len(varname_list)>1:
            raise ValueError('There seems to be an inconsistency')
        varname = varname_list[0]
        print cellname, varname
        currlist.append(varname)
    dict_hl_warmup_cells[sectorname] = currlist

    
    

