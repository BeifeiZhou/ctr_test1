#This script is used to transform the raw data to the GBDT input data

import argparse, csv, sys, os
print 'Usage: python pre_gbdt.py 20'
print '20 is the number of target features'

target_num = sys.argv[1]

#Choose target features from the counting result
cmd = 'tail -'+target_num+' ../input/count10.txt | awk -F \',\' \'{print $1","$2}\' > ../input/target_features.txt'
os.system(cmd)

#Get target features
target_features = open('../input/target_features.txt','r').readlines()
target_features = map(lambda x: x.strip(), target_features)

feature_names = 'Campaign ID\tUser ID\tTimestamp\tCity\tBrowser\tOS\tDevice\tDomain'.split('\t')
#Transform each activity into a number 0-3
dict_act = {'impression':'0','click':'1','retargeting':'2','conversion':'3'}
#Match each entry with the target features
with open('../input/gbdt_input.csv', 'w') as f_s:
    for row in csv.DictReader(open('../input/dataset.csv'), delimiter='\t'):
        cat_feats = set()
        for field in feature_names:
            key = field + ',' + row[field]
            cat_feats.add(key)

        feats = []
        for j, feat in enumerate(target_features, start=1):
            if feat in cat_feats:
                feats.append("1")
            else:
                feats.append("0")
        f_s.write(','.join(feats) + ',' + dict_act[row['Activity']]+'\n')
