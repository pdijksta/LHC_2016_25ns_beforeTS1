import numpy as np
import pickle, sys


def reconfigure_pickle(picklename, fillnumbers):

    with open(picklename, 'rb') as fid:
        fill_dict = pickle.load(fid)

    fillnumbers = np.atleast_1d(fillnumbers)
    for fill in fillnumbers:
        fill_dict[int(fill)] = 'incomplete'

    with open(picklename, 'wb') as fid:
        pickle.dump(fill_dict, fid)

if len(sys.argv)>2:
    picklename = sys.argv[1]
    fillnumbers = sys.argv[2].split(',')
    reconfigure_pickle(picklename, fillnumbers)
else:
    print "Provide picklename and list of fills!"
