import cPickle

# To be tested!

file1 = ''
file2 = ''
outfile = ''

with open(file1, 'r') as f:
    dict1 = cPickle.load(f)
with open(file2, 'r') as f:
    dict2 = cPickle.load(f)


filln1 = dict1['filln']
filln2 = dict2['filln']

for index2, filln in enumerate(filln2):
    filln1.append(filln)
    filln1.sort()
    index1 = filln1.index(filln)
    for key, item in dict2.iteritems():
        if key is not 'filln':
            to_insert = dict2[key][index2]
            if hasattr(dict1[key], 'insert'):
                if key in dict1:
                    dict1[key].insert(index1, to_insert)
                else:
                    print('dict1 does not have key %s!' % key)
            else:
                print('Key %s is not iterable!' % key)

with open(outfile, 'w') as f:
    cPickle.dump(dict1, f, -1)
