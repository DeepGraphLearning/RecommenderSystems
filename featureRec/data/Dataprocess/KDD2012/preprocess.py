#coding=utf-8
#Email of the author: zjduan@pku.edu.cn
'''
0. Click：
1. Impression（numerical）
2. DisplayURL: (categorical)
3. AdID:(categorical) 
4. AdvertiserID:(categorical)
5. Depth:(numerical)
6. Position:(numerical)
7. QueryID:  (categorical) the key of the data file 'queryid_tokensid.txt'. 
8. KeywordID: (categorical)the key of  'purchasedkeyword_tokensid.txt'.
9. TitleID:  (categorical)the key of 'titleid_tokensid.txt'.
10. DescriptionID:  (categorical)the key of 'descriptionid_tokensid.txt'.
11. UserID: (categorical)the key of 'userid_profile.txt'
12. User's Gender: (categorical)
13. User's Age: (categorical)
'''
import math
train_path = './training.txt'
f1 = open(train_path, 'r')
f2 = open('./userid_profile.txt', 'r')
dic = {}
f_train_value = open('./train_x.txt', 'w')
f_train_index = open('./train_i.txt', 'w')
f_train_label = open('./train_y.txt', 'w')
debug = False
tune = False
Column = 12
Field = 13

numr_feat = [1,5,6]
numerical = [0] * Column
cate_feat = [2,3,4,7,8,9,10,11]
index_cnt = 0 
index_others = [0] * (Field + 1)
Max = [0] * 12
numerical[0] = -1
for i in numr_feat:
    index_others[i] = index_cnt
    index_cnt += 1
    numerical[i] = 1
for i in cate_feat:
    index_others[i] = index_cnt
    index_cnt += 1

for i in range(Field + 1):
    dic[i] = dict()

###init user_dic
user_dic = dict()

cnt_line = 0
for line in f2:
    cnt_line += 1
    if (cnt_line % 1000000 == 0):
        print ("cnt_line = %d, index_cnt = %d" % (cnt_line, index_cnt))
    # if (debug == True):
    #     if (cnt_line >= 10000):
    #         break
    split = line.strip('\n').split('\t')
    user_dic[split[0]] = [split[1], split[2]]
    if (split[1] not in dic[12]):
        dic[12][split[1]] = [index_cnt, 0]
        index_cnt += 1
    if (split[2] not in dic[13]):
        dic[13][split[2]] = [index_cnt, 0]
        index_cnt += 1

cnt_line = 0
for line in f1:
    cnt_line += 1
    if (cnt_line % 1000000 == 0):
        print ("cnt_line = %d, index_cnt = %d" % (cnt_line, index_cnt))
    if (debug == True):
        if (cnt_line >= 10000):
            break
    split = line.strip('\n').split('\t')
    for i in cate_feat:
        if (split[i] != ''):
            if split[i] not in dic[i]:
                dic[i][split[i]] = [index_others[i], 0]
            dic[i][split[i]][1] += 1
            if (dic[i][split[i]][0] == index_others[i] and dic[i][split[i]][1] == 10):
                dic[i][split[i]][0] = index_cnt
                index_cnt += 1

    if (tune == False):
        label = split[0]
        if (label != '0'): label = '1'
        index = [0] * Field
        value = ['0'] * Field
        for i in range(1, 12):
            if (numerical[i] == 1):
                index[i - 1] = index_others[i]
                if (split[i] != ''):
                    value[i - 1] = split[i]
                    Max[i] = max(int(split[i]), Max[i])
            else:
                if (split[i] != ''):
                    index[i - 1] = dic[i][split[i]][0]
                    value[i - 1] = '1'
                    
            if (split[i] == ''):
                value[i - 1] = '0'  
            if (i == 11 and split[i] == '0'):
                value[i - 1] = '0'
        ### gender and age
        if (split[11] == '' or (split[11] not in user_dic)):
            index[12 - 1] = index_others[12]
            value[12 - 1] = '0'
            index[13 - 1] = index_others[13]
            value[13 - 1] = '0'
        else:
            index[12 - 1] = dic[12][user_dic[split[11]][0]][0]
            value[12 - 1] = '1'
            index[13 - 1] = dic[13][user_dic[split[11]][1]][0]
            value[13 - 1] = '1'

        f_train_index.write(' '.join(str(i) for i in index) + '\n')
        f_train_value.write(' '.join(value) + '\n')
        f_train_label.write(label + '\n')

f1.close()
f_train_index.close()
f_train_value.close()
f_train_label.close()
print ("Finished!")
print ("index_cnt = %d" % index_cnt)
print ("max number for numerical features:")
for i in numr_feat:
    print ("no.:%d max: %d" % (i, Max[i]))
