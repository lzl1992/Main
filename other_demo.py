import pandas as pd
import numpy as np
from collections import Counter
import sys
import input

table_1 = pd.read_csv("C:/2019Summer/phasing_opt/output/chr22.compare.haps",
                      sep=' ', header=None, names=None, index_col=None, engine='python')
# table_2 = pd.read_csv('C:\\2019Summer\\Phasing Project\\Output\\chr22_500iter.prephase.haps.backup',
                     # sep=' ', header=None, names=None, index_col=None, engine='python')
t1 = table_1.iloc[:, -2]
t2 = table_1.iloc[:, -1]
LOCUS_ID = np.array(table_1.iloc[:, 2])

compare1 = np.array(t1)
# compare1.reshape((8512, 1))
# compare1.transpose()
compare2 = np.array(t2)
# compare2.reshape((8512, 1))
# compare2.transpose()
#t2 = table_2.drop(columns=[0, 1, 2, 3, 4], inplace=False)
Compare_G = np.add(compare1, compare2)

my_h = pd.read_csv("C:/2019Summer/phasing_opt/output/result_demo.txt",
                   sep=' ', header=None, names=None, index_col=None, engine='python')
my_h = np.array(my_h)
h1 = my_h[0, :]
h1.reshape((8512, 1))
h2 = my_h[1, :]
h2.reshape((8512, 1))
G = np.add(h1, h2)

Ne150 = pd.read_csv("C:/2019Summer/phasing_opt/output/result_Ne150.txt",
                    sep=' ', header=None, names=None, index_col=None, engine='python')
h1_150 = np.array(Ne150)[0, :]
h2_150 = np.array(Ne150)[1, :]
G_150 = np.add(h1_150, h2_150)
'''
def cal_diff(x, y):
    assert isinstance(x, int)
    a = table_1.iloc[:, x]
    b = h1
    diff = 0
    for i in range(0, len(b)-1):
        if a.loc[i] == b[i]:
            diff = diff
        else:
            diff += 1
    return diff
'''
s, h, r, s_ID = input.get_input()
s_ID = np.array(s_ID)
answer1 = h[:, 2]
answer2 = h[:, 3]
answer_G = np.add(answer1, answer2)


def locus_diff(my_id, his_id):
    different = []
    for i in range(len(my_id)):
        if int(my_id[i]) != his_id[i]:
            different.append(i)
    np.array(different)
    return different


LOCUS_DIFF = locus_diff(s_ID, LOCUS_ID)


def cal_diff(student, student_G, correct, correct_G):
    diff = 0
    heter_count = 0
    total_diff = 0
    seg_info = []
    student_seg_info = []
    seg_index = []
    student_seg_index = []
    temp = []
    student_temp = []
    cut = 0
    print(student.shape[0])
    diff_position = []
    for i in range(student.shape[0]):
        if student[i] != correct[i] and student_G[i] == 1 and correct_G[i] == 1:
            diff += 1
        if student_G[i] == 1 and correct_G[i] == 1:
            heter_count += 1
        if student_G[i] != correct_G[i]:
            diff_position.append(i)
            total_diff += 1
        if correct_G[i] == 1 and student_G[i] == 1:
            temp.append(correct[i])
            student_temp.append(student[i])
            cut += 1
            if cut == 3:
                seg_info.append(temp)
                student_seg_info.append(student_temp)
                temp = []
                student_temp = []
                cut = 0
    diff_position = np.array(diff_position)
    for x in seg_info:
        seg_index.append(4*x[0]+2*x[1]+1*x[2])
    seg_count = 0
    seg_diff = 0
    for y in student_seg_info:
        student_seg_index.append(4*y[0]+2*y[1]+1*y[2])
        if student_seg_index[seg_count] != seg_index[seg_count]:
            seg_diff += 1
        seg_count += 1
    count = Counter(seg_index)
    print(diff_position)
    diff_his = []
    for x in diff_position:
        diff_his.append(correct_G[x])
    print(diff_his)
    print(total_diff)
    myphase = np.array(student_seg_index).T
    hisphase = np.array(seg_index).T
    bias = 0
    bias_information = []
    for i in range(len(hisphase)):
        if myphase[i] != hisphase[i] and myphase[i] + hisphase[i] != 7:
            bias += 1
            bias_information.append(i)
    bias_information = np.array(bias_information)
    print(myphase)
    print(hisphase)
    with open(sys.path[0]+'/output/bias_information_change_distance.txt', 'w') as f:
        f.write(' '.join([str(x) for x in bias_information])+'\n')
    return diff, heter_count


print(cal_diff(h1, G, answer1, answer_G))
'''
m = []
for i in range(0, 24):
    print(i)
    x = 5
    y = 1
    d1 = cal_diff(x, x)+cal_diff(y, y)
    d2 = cal_diff(x, y)+cal_diff(y, x)
    m.append(min(d1, d2))
print(m)
total = sum(m)
'''
