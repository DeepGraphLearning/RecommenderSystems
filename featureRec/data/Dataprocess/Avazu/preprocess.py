#coding=utf-8
#Email of the author: zjduan@pku.edu.cn
'''
0.id: ad identifier
1.click: 0/1 for non-click/click
2.hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
3.C1 -- anonymized categorical variable
4.banner_pos
5.site_id
6.site_domain
7.site_category
8.app_id
9.app_domain
10.app_category
11.device_id
12.device_ip
13.device_model
14.device_type
15.device_conn_type
16.C14
17.C15
18.C16
19.C17
20.C18
21.C19
22.C20
23.C21
'''
import pandas as pd
import math
train_path = './train.csv'
f1 = open(train_path, 'r')
dic = {}
f_train_value = open('./train_x.txt', 'w')
f_train_index = open('./train_i.txt', 'w')
f_train_label = open('./train_y.txt', 'w')
debug = False
tune = False
Bound = [5] * 24

label_index = 1
Column = 24

numr_feat = []
numerical = [0] * Column
numerical[label_index] = -1

cate_feat = []
for i in range(Column):
    if (numerical[i] == 0):
        cate_feat.extend([i])

index_cnt = 0 
index_others = [0] * Column


for i in numr_feat:
    index_others[i] = index_cnt
    index_cnt += 1
    numerical[i] = 1
for i in cate_feat:
    index_others[i] = index_cnt
    index_cnt += 1

for i in range(Column):
    dic[i] = dict()

cnt_line = 0
for line in f1:
    cnt_line += 1
    if (cnt_line == 1): continue # header
    if (cnt_line % 1000000 == 0):
        print ("cnt_line = %d, index_cnt = %d" % (cnt_line, index_cnt))
    if (debug == True):
        if (cnt_line >= 10000):
            break
    split = line.strip('\n').split(',')
    for i in cate_feat:
        if (split[i] != ''):
            if split[i] not in dic[i]:
                dic[i][split[i]] = [index_others[i], 0]
            dic[i][split[i]][1] += 1
            if (dic[i][split[i]][0] == index_others[i] and dic[i][split[i]][1] == Bound[i]):
                dic[i][split[i]][0] = index_cnt
                index_cnt += 1
                
    if (tune == False):
        label = split[label_index]
        if (label != '0'): label = '1'
        index = [0] * (Column - 1)
        value = ['0'] * (Column - 1)
        for i in range(Column):
            cur = i
            if (i == label_index): continue
            if (i > label_index): cur = i - 1
            if (numerical[i] == 1):
                index[cur] = index_others[i]
                if (split[i] != ''):
                    value[cur] = split[i]
            else:
                if (split[i] != ''):
                    index[cur] = dic[i][split[i]][0]
                    value[cur] = '1'
                    
            if (split[i] == ''):
                value[cur] = '0'  

        f_train_index.write(' '.join(str(i) for i in index) + '\n')
        f_train_value.write(' '.join(value) + '\n')
        f_train_label.write(label + '\n')

f1.close()
f_train_index.close()
f_train_value.close()
f_train_label.close()
print ("Finished!")
print ("index_cnt = %d" % index_cnt)



