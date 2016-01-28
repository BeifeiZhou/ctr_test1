#This script is used to count the number of activities for each value in each feature

import argparse, csv, sys, collections, os
print 'Usage: python count.py'

#Add one row at the begining to specify the names of each column
cmd = "echo 'Campaign ID\tUser ID\tTimestamp\tCity\tBrowser\tOS\tDevice\tDomain\tActivity' > tmp | cat tmp ../input/affectv_ds_test_data > ../input/dataset.csv"
os.system(cmd)
cmd = 'rm tmp'
os.system(cmd)

#bulid a count dictionary to store the count result for each type of activities
counts = collections.defaultdict(lambda : [0, 0, 0, 0, 0])

#Counting
feature_names = 'Campaign ID\tUser ID\tTimestamp\tCity\tBrowser\tOS\tDevice\tDomain'.split('\t')
for i, row in enumerate(csv.DictReader(open('../input/dataset.csv'), delimiter='\t'), start=1):
    label = row['Activity']
    for field in feature_names:
        value = row[field]
        if label == 'impression':
            counts[field+','+value][0] += 1
        elif label == 'click':
            counts[field+','+value][1] += 1
        elif label == 'retargeting':
            counts[field+','+value][2] += 1
        else:
            counts[field+','+value][3] += 1
        counts[field+','+value][4] += 1
    if i % 1000000 == 0:
        sys.stderr.write('{0}m\n'.format(int(i/1000000)))

#Write the counting result
count = open('../input/count10.txt', 'w')
count.write('field,value,impression,click,retargeting,conversion,total\n')
for key, (imp, click, retar, con, total) in sorted(counts.items(), key=lambda x: x[1][4]):
    if total < 10:
        continue
    line = key+','+str(imp)+','+str(click)+','+str(retar)+','+str(con)+','+str(total)
    count.write(line+'\n')
