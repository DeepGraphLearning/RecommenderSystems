dict = {}

user_count = 6040

gender = {}
gender['M'] = 1
gender['F'] = 2

dict[1] = "Gender-male"
dict[2] = "Gender-female"

age = {}
age['1'] = 3
age['18'] = 4
age['25'] = 5
age['35'] = 6
age['45'] = 7
age['50'] = 8
age['56'] = 9

dict[3] = "Age-under 18"
dict[4] = "Age-18-24"
dict[5] = "Age-25-34"
dict[6] = "Age-35-44"
dict[7] = "Age-45-49"
dict[8] = "Age-50-55"
dict[9] = "Age-56+"

feature_size = 9
occ = {}
for i in range(21):
	feature_size += 1
	occ[str(i)] = feature_size

dict[10] = "Occupation-other"
dict[11] = "Occupation-academic/educator"
dict[12] = "Occupation-artist"
dict[13] = "Occupation-clerical/admin"
dict[14] = "Occupation-college/grad student"
dict[15] = "Occupation-customer service"
dict[16] = "Occupation-doctor/health care"
dict[17] = "Occupation-executive/managerial"
dict[18] = "Occupation-farmer"
dict[19] = "Occupation-homemaker"
dict[20] = "Occupation-K-12 student"
dict[21] = "Occupation-lawyer"
dict[22] = "Occupation-programmer"
dict[23] = "Occupation-retired"
dict[24] = "Occupation-sales/marketing"
dict[25] = "Occupation-scientist"
dict[26] = "Occupation-self-employed"
dict[27] = "Occupation-technician/engineer"
dict[28] = "Occupation-tradesman/craftsman"
dict[29] = "Occupation-unemployed"
dict[30] = "Occupation-writer"

f = open('users.dat', 'r')
zipcode = {}
for i in range(1, user_count + 1):
	line = f.readline()
	line = line[:-1]
	l = line.split('::')
	if zipcode.get(l[-1]) == None:
		feature_size += 1
		zipcode[l[-1]] = feature_size
		dict[feature_size] = "Zipcode-" + str(l[-1])
f.close()

f = open('users.dat', 'r')
user_i = [[]]
user_v = [[]]
for i in range(1, user_count + 1):
	line = f.readline()
	line = line[:-1]
	l = line.split('::')
	user_i.append([gender[l[1]], age[l[2]], occ[l[3]], zipcode[l[4]]])
	user_v.append([1, 1, 1, 1])
f.close()
print("The number of user's feature is:", len(user_i))

movie_count = 3883

max_gen = 0
min_gen = 10

year_dict = {}
for i in range(1919, 1930):
	year_dict[i] = 1
for i in range(1930, 1940):
	year_dict[i] = 2
for i in range(1940, 1950):
	year_dict[i] = 3
for i in range(1950, 1960):
	year_dict[i] = 4
for i in range(1960, 1970):
	year_dict[i] = 5
for i in range(1970, 2001):
	year_dict[i] = 6 + i - 1970

f = open('movies.dat', 'r', encoding="ISO-8859-1")
genres = {}
for i in range(1, movie_count + 1):
	line = f.readline()
	line = line[:-1]
	l = line.split('::')
	s = l[-1]
	l = s.split('|')
	if len(l) > max_gen:
		max_gen = len(l)
	if len(l) < min_gen:
		min_gen = len(l)
	if len(l) == 0:
		print('error')
	for _ in l:
		if genres.get(_) == None:
			feature_size += 1
			genres[_] = feature_size
			dict[feature_size] = "Genre-" + _
f.close()
print("2222", feature_size)
print(len(dict))

print('The max number is :', max_gen)

#feature_size += 1 # for year of release

f = open('movies.dat', 'r', encoding="ISO-8859-1")
movie_i = {}
movie_v = {}
for i in range(1, movie_count + 1):
	line = f.readline()
	line = line[:-1]
	l = line.split('::')
	MovieID = int(l[0])
	Year = int(l[1][-5:-1])
	l = l[-1].split('|')
	new_i = []
	new_v = []
	for _ in l:
		new_i.append(genres[_])
		new_v.append(1)
	t = 6 - len(l) # 0 ~ 5 remain
	for _ in range(feature_size + 1, feature_size + t + 1):
		new_i.append(_)
		new_v.append(0)
	#new_i.append(feature_size + 6)
	#new_v.append(Year)
	new_i.append(feature_size + 5 + year_dict[Year])
	new_v.append(1)
	movie_i[MovieID] = new_i
	movie_v[MovieID] = new_v
f.close()

print(feature_size + 1, feature_size + 5)
#feature_size += 6
dict[feature_size + 1] = "Genre-NULL"
dict[feature_size + 2] = "Genre-NULL"
dict[feature_size + 3] = "Genre-NULL"
dict[feature_size + 4] = "Genre-NULL"
dict[feature_size + 5] = "Genre-NULL"
feature_size += 5


feature_size += 1
dict[feature_size] = "Release-1919-1929"
feature_size += 1
dict[feature_size] = "Release-1930-1939"
feature_size += 1
dict[feature_size] = "Release-1940-1949"
feature_size += 1
dict[feature_size] = "Release-1950-1959"
feature_size += 1
dict[feature_size] = "Release-1960-1969"
for y in range(1970, 2001):
	feature_size += 1
	dict[feature_size] = "Release-" + str(y)

print("####: ", feature_size)
print(len(dict))

print("The number of movie's feature is:", len(movie_i))

feature_size += 1 # for timestamp
dict[feature_size] = "Timestamp"

f = open('ratings.dat', 'r')

data_i = []
data_v = []
Y = []
#U = []
#I = []
all_count = 1000209
ratings_count = 0
for i in range(1, all_count + 1):
	line = f.readline()
	line = line[:-1]
	l = line.split('::')
	y = int(l[2])
	new_i = user_i[int(l[0])].copy()
	new_v = user_v[int(l[0])].copy()
	new_i.extend(movie_i[int(l[1])])
	new_v.extend(movie_v[int(l[1])])
	new_i.append(feature_size)
	new_v.append(int(l[3]))
	if y > 3:
		y = 1
	elif y < 3:
		y = 0
	else:
		y = -1

	if y != -1:
		data_i.append(new_i)
		data_v.append(new_v)
	#	U.append(int(l[0]))
	#	I.append(int(l[1]))
		Y.append(y)
		ratings_count += 1
f.close()
print('valid number of ratings:', len(data_v))

print('Positive number =', sum(Y))
print(feature_size)
print("Dict: ", len(dict))
print('All =', len(data_i))

import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
R = []
for i in range(ratings_count):
	R.append([data_v[i][-1]])
#print(R)
#print(np.max(R))
#print(np.min(R))
R = scaler.fit_transform(R)
#print(R)

for i in range(ratings_count):
	data_v[i].pop()
	data_v[i].append(R[i][0])
#	data_v[i].append(U[i])
#	data_v[i].append(I[i])
print(data_v[0])
perm = []
for i in range(ratings_count):
	perm.append(i)
        
random.seed(2019)        
random.shuffle(perm)

train_count = int(ratings_count * 0.8)
valid_count = int(ratings_count * 0.9)
X_i_tr = []
X_v_tr = []
Y_tr = []
for i in range(train_count):
	X_i_tr.append(data_i[perm[i]])
	X_v_tr.append(data_v[perm[i]])
	Y_tr.append(Y[perm[i]])

X_i_tr = np.array(X_i_tr)
X_v_tr = np.array(X_v_tr)
Y_tr = np.array(Y_tr)


i1 = X_i_tr[:, 0:4]
i2 = X_i_tr[:, 4:10]
i3 = X_i_tr[:, 10:]
x1 = X_v_tr[:, 0:4]
x2 = X_v_tr[:, 4:10]
x3 = X_v_tr[:, 10:]
i4 = np.concatenate((i1,i3), axis=1)
x4 = np.concatenate((x1,x3), axis=1)

np.save("train_i_genre.npy", i2)
np.save("train_i_other.npy", i4)
np.save("train_x_genre.npy", x2)
np.save("train_x_other.npy", x4)
np.save("train_y.npy", np.array(Y_tr))
#np.save("train_ui.npy", np.array(ui_tr))
X_i_va = []
X_v_va = []
Y_va = []
for i in range(train_count, valid_count):
	X_i_va.append(data_i[perm[i]])
	X_v_va.append(data_v[perm[i]])
	Y_va.append(Y[perm[i]])
#	ui_va.append([U[perm[i]], I[perm[i]])
X_i_va = np.array(X_i_va)
X_v_va = np.array(X_v_va)
Y_va = np.array(Y_va)

i1 = X_i_va[:, 0:4]
i2 = X_i_va[:, 4:10]
i3 = X_i_va[:, 10:]
x1 = X_v_va[:, 0:4]
x2 = X_v_va[:, 4:10]
x3 = X_v_va[:, 10:]
i4 = np.concatenate((i1,i3), axis=1)
x4 = np.concatenate((x1,x3), axis=1)

np.save("valid_i_genre.npy", i2)
np.save("valid_i_other.npy", i4)
np.save("valid_x_genre.npy", x2)
np.save("valid_x_other.npy", x4)
np.save("valid_y.npy", np.array(Y_va))



X_i_te = []
X_v_te = []
Y_te = []
for i in range(valid_count, ratings_count):
	X_i_te.append(data_i[perm[i]])
	X_v_te.append(data_v[perm[i]])
	Y_te.append(Y[perm[i]])
#	ui_te.append(U[perm[i]]], I[perm[i]])


X_i_te = np.array(X_i_te)
X_v_te = np.array(X_v_te)
Y_te = np.array(Y_te)


i1 = X_i_te[:, 0:4]
i2 = X_i_te[:, 4:10]
i3 = X_i_te[:, 10:]
x1 = X_v_te[:, 0:4]
x2 = X_v_te[:, 4:10]
x3 = X_v_te[:, 10:]
i4 = np.concatenate((i1,i3), axis=1)
x4 = np.concatenate((x1,x3), axis=1)

np.save("test_i_genre.npy", i2)
np.save("test_i_other.npy", i4)
np.save("test_x_genre.npy", x2)
np.save("test_x_other.npy", x4)
np.save("test_y.npy", np.array(Y_te))


print(len(X_i_tr))
print(len(X_i_va))
print(len(X_i_te))
print(len(Y))

f = open("feature.txt", 'w')
f.write(str(dict))
