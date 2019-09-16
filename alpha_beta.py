import test
from math import log10
import input
from sklearn.preprocessing import normalize
from random import randrange
import multiprocessing as mp


def init_prob(first, reference, K):
    """
    :param first: the key at first position
    :param reference: Hg
    :param K: population
    :return: initial distribution, pi
    """
    now_dict = reference.find_segment(0).haplotype_link
    if first in now_dict:
        p = now_dict.get(first)/K
    else:
        p = 0
    return p


def emission_prob(observe, hidden, theta1, theta2, heterozygous):
    """

    :param observe: observation value
    :param hidden: hidden value
    :param theta1: if same
    :param theta2: if different
    :return: emission probability
    """
    if observe == hidden:
        return theta1 / heterozygous
    else:
        return theta2 / heterozygous


def inner_transition_prob(now, now_value, now_dict, next, next_value, next_dict, m, thou1, thou2, reference, K):
    """

    :param now: state at m
    :param now_value: weight of now state
    :param now_dict: weight database at m
    :param next: state at (m+1)
    :param next_value: weight of next state
    :param next_dict: weight database at (m+1)
    :param m: position
    :param thou1: recombination
    :param thou2: Non-recombination
    :param reference: Hg
    :param K: population

    :return: one step transition prob from now state to next state (Inner-segment)
    """
    if now == next:
        p = thou2 + (thou1 * next_value) / K
    else:
        p = thou1 * next_value / K
    return p


def inter_transition_prob(now, now_value, now_dict, next, next_value, next_dict, next_seg_idx, m, thou1, thou2, reference, K, Pivot, trans_dict):
    """

    :param now: state at m
    :param now_value: weight of now state
    :param now_dict: weight database at m
    :param next: state at (m+1)
    :param next_value: weight of next state
    :param next_dict: weight database at (m+1)
    :param next_seg_idx: next segment index
    :param m: position
    :param thou1: recombination
    :param thou2: Non-recombination
    :param reference: Hg
    :param K: population
    :param Pivot: Pivot position
    :param trans_dict: transition database between m and (m+1)

    :return: one step transition prob from now state to next state (Inter-segment)
    """
    # 顺序的区别来源于之前压缩矩阵过程，如果该transition存在于transition字典中则进行第一种计算
    if m < Pivot-1 and (next, now) in trans_dict:
        p = thou2 * trans_dict.get((next, now)) / now_value + thou1 * next_value / K
    elif m >= Pivot-1 and (now, next) in trans_dict:
        p = thou2 * trans_dict.get((now, next)) / now_value + thou1 * next_value / K
    # 如果不在字典中则进行另一种计算
    else:
        p = thou1 * next_value / K
    return p


def index_branch(m, start_index, individual, Pivot):    # 对应forward计算
    """

    :param m: position
    :param start_index: Hg cut
    :param individual: Sg
    :param Pivot: Pivot position
    :return: branch of m, decide what to be calculated later
    """
    if m == 0:  # 第一行
        branch = 0
    elif m == Pivot:    # 枢纽点必是Hg cut
        if m in individual.start_index:     # 如果还有Sg分段
            branch = 4
        else:   # 如果只是Hg分段
            branch = 2
    elif m in individual.start_index:   # Sg分段
        if m not in start_index:    # 如果只是Sg分段
            branch = 1
        else:   # 如果既是Sg分段又是Hg分段
            branch = 3
    elif m in start_index:  # Hg分段
        branch = 2
    else:   # 普通情况，什么都没有
        branch = 5
    return branch


def backward_index_branch(m, backward, individual, Pivot):  # 由于是Backward计算，所以采取 -1 处理
    """

    :param m: position
    :param backward: Hg cut
    :param individual: Sg
    :param Pivot: Pivot position
    :return: branch of m, decide what to be calculated later
    """
    if m == Pivot-1:    # 枢纽点必是Hg cut
        if m in individual.backward:    # 如果还是Sg cut
            branch = 4
        else:
            branch = 2
    elif m in individual.backward:  # Sg cut
        if m not in backward:  # 如果不是Hg cut
            branch = 1
        else:   # 既是Sg cut又是Hg cut
            branch = 3
    elif m in backward:     # Hg cut
        branch = 2
    elif m == individual.matrix_sg.shape[0]-1:  # 最后一行
        branch = -1
    else:   # 普通情况，什么也不是
        branch = 5
    return branch


def normalize(m, cube):
    """
    This function is for normalize alpha or beta of every m, in order to avoid the underflow problem
    :param m: position
    :param cube: alpha or beta
    :return: the normalize result
    """
    face = cube[:, :, m]
    face_sum = test.np.sum(face)
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            cube[i][j][m] = cube[i][j][m] / face_sum


def cal_alpha(thou, reference, individual, K, theta, J, Pivot):
    """

    :param thou: recombination parameter
    :param reference: compacted hg
    :param individual: compacted sg
    :param K: total population
    :param theta: mutation parameter
    :return: alpha cube, which represents the forward variables
    """
    # print("start calculate alpha!")
    x = individual.matrix_sg.shape[1]  # initial the x-axis length(the type number in sg, pow(2,3)=8 normally)
    y = J   # initial y-axis length(the type number upperbound in hg)
    z = individual.matrix_sg.shape[0]   # initial z-axis length(total locus number)
    alpha = test.np.zeros((x, y, z))
    for segment in reference.segments:
        reference.start_index.append(segment.start_position)
    theta1 = (2*K+theta)/(2*K+2*theta)
    theta2 = 1-theta1
    trans_occur = [2, 3, 4]
    trans_dict = None
    for m in range(z):
        if individual.heterozygous[m] is True:
            heterozygous = x / 2
        else:
            heterozygous = x
        if m > 0:   # 拿到重组参数thou1和thou2
            thou1 = thou[m-1]
            thou2 = 1-thou1
        else:
            thou1 = thou2 = 0
        if m >= 1:  # in this part, pre means m-1 position
            pre_dict = reference.find_segment(reference.seg_idx_dict.get(m-1)).haplotype_link   # 求(m-1)位置dict
        else:
            pre_dict = reference.find_segment(0).haplotype_link
        now_segment_idx = reference.seg_idx_dict.get(m)
        now_segment = reference.find_segment(now_segment_idx)
        now_dict = now_segment.haplotype_link  # 拿到 m 位置dict
        if m < z-1:
            next_segment_idx = reference.seg_idx_dict.get(m+1)
            next_dict = reference.find_segment(next_segment_idx).haplotype_link  # 求(m+1)位置dict
        else:
            next_segment_idx = now_segment_idx
            next_dict = now_dict
        m_position = m - now_segment.start_position  # m距离当前段首的距离
        branch = index_branch(m, reference.start_index, individual, Pivot)
        if branch in trans_occur:   # 若发生Hg transition，则求 trans dict
            if m < Pivot:
                trans_dict = reference.find_transition(next_segment_idx - 1).transition
            elif m == Pivot:
                trans_dict = reference.find_transition(reference.total_transitions()-1).transition
                trans_dict = trans_dict[0]
            else:   # 由于Pivot transition被放在了最后
                '''测试用的是-1，真实情况是-2，记得回来改'''
                trans_dict = reference.find_transition(next_segment_idx - 2).transition
        if branch == 0:     # 第一行
            for i in range(x):   # 每一个i
                column = 0
                observed = individual.matrix_sg[m][i]   # 观测值，Xm
                """由于每一种reference中的基因数据都被存在Hash表中，所以遍历哈希表，同时用column来记录当前index"""
                for j, values in now_dict.items():
                    hidden = now_segment.matrix_form[m_position][column][0]     # 隐藏变量，Zm。第三个0是求具体基因信息
                    emi_prob = emission_prob(observed, hidden, theta1, theta2, heterozygous)   # bm（Oi），观测概率
                    alpha[i][column][m] = emi_prob * init_prob(j, reference, K)    # alpha_0(i,j) 的初值公式
                    column += 1
        elif branch == 5:   # 普通情况
            for i in range(x):
                weight = []
                current_alpha = alpha[i, :, m-1]
                current_sum = test.np.sum(current_alpha)
                current_delta = []
                for s in range(J - 1):
                    current_delta.append(current_alpha[s + 1] - current_alpha[s])
                column = 0
                pre_sum = 0
                temp_sum = 0
                observed = individual.matrix_sg[m][i]
                for j, values in now_dict.items():
                    hidden = now_segment.matrix_form[m_position][column][0]
                    emi_prob = emission_prob(observed, hidden, theta1, theta2, heterozygous)
                    weight.append(values)
                    if column == 0:
                        v = 0
                        for temp_j, temp_values in pre_dict.items():
                            temp_sum += alpha[i][v][m - 1] * inner_transition_prob(temp_j, temp_values, pre_dict, j, values, now_dict, m - 1, thou1, thou2, reference, K)
                            v += 1
                        pre_sum = temp_sum
                        alpha[i][column][m] = emi_prob * temp_sum
                    else:
                        delta = current_sum * (weight[column] - weight[column - 1]) * thou1 / K + thou2 * current_delta[column - 1]
                        temp_sum = delta + pre_sum
                        pre_sum = temp_sum
                        alpha[i][column][m] = emi_prob * temp_sum
                    column += 1
        elif branch == 1:   # Sg cut
            temp_sum = [0]
            weight = []
            current_alpha = alpha[:, :, m - 1]
            current_sum = test.np.sum(current_alpha, axis=(0, 1))
            current_delta = []
            for j in range(J - 1):
                alpha_mij1 = alpha[:, j+1, m-1]
                alpha_mij = alpha[:, j, m-1]
                current_delta.append(test.np.sum(alpha_mij1) - test.np.sum(alpha_mij))
            for i in range(x):
                column = 0
                observed = individual.matrix_sg[m][i]
                for j, values in now_dict.items():
                    hidden = now_segment.matrix_form[m_position][column][0]
                    emi_prob = emission_prob(observed, hidden, theta1, theta2, heterozygous)
                    if i == 0:
                        weight.append(values)
                        if column == 0:
                            for u in range(x):
                                v = 0
                                for temp_j, temp_values in pre_dict.items():
                                    temp_sum[column] += alpha[u][v][m-1] * inner_transition_prob(temp_j, temp_values, pre_dict, j, values, now_dict, m-1, thou1, thou2, reference, K)
                                    v += 1
                                alpha[i][column][m] = emi_prob * temp_sum[column]
                        else:
                            delta = current_sum * thou1 * (weight[column] - weight[column-1]) / K + thou2 * current_delta[column-1]
                            temp_sum.append(delta + temp_sum[column-1])
                            alpha[i][column][m] = emi_prob * temp_sum[column]
                    else:
                        alpha[i][column][m] = emi_prob * temp_sum[column]
                    column += 1
            weight.clear()
            current_delta.clear()
            temp_sum.clear()
        elif branch == 2:   # Hg cut 以及普通枢纽点(普通枢纽点就是Hg cut)
            for i in range(x):
                column = 0
                observed = individual.matrix_sg[m][i]
                for j, values in now_dict.items():
                    hidden = now_segment.matrix_form[m_position][column][0]
                    emi_prob = emission_prob(observed, hidden, theta1, theta2, heterozygous)
                    temp_sum = 0
                    v = 0
                    for temp_j, temp_values in pre_dict.items():
                        temp_sum += alpha[i][v][m - 1] * inter_transition_prob(temp_j, temp_values, pre_dict, j, values, now_dict, next_segment_idx, m - 1, thou1, thou2, reference, K, Pivot, trans_dict)
                        v += 1
                    alpha[i][column][m] = emi_prob * temp_sum
                    column += 1
        else:  # Hg and Sg cut 以及 枢纽点 and Sg cut， 由于trans_dict来自之前的计算，所以可以自动区分，其他计算逻辑没有区别
            for i in range(x):
                column = 0
                observed = individual.matrix_sg[m][i]
                for j, values in now_dict.items():
                    hidden = now_segment.matrix_form[m_position][column][0]
                    emi_prob = emission_prob(observed, hidden, theta1, theta2, heterozygous)
                    temp_sum = 0
                    for u in range(x):
                        v = 0
                        for temp_j, temp_values in pre_dict.items():
                            temp_sum += alpha[u][v][m - 1] * inter_transition_prob(temp_j, temp_values, pre_dict, j, values, now_dict, next_segment_idx, m - 1, thou1, thou2, reference, K, Pivot, trans_dict)
                            v += 1
                    alpha[i][column][m] = emi_prob * temp_sum
                    column += 1
        normalize(m, alpha)
    return alpha


def cal_beta(thou, reference, individual, K, theta, J, Pivot):
    """

    :param thou: recombination parameter
    :param reference: Hg
    :param individual: Sg
    :param K: population
    :param theta: mutation parameter
    :param J: type limit
    :param Pivot: start position
    :return: beta cube, which represents the backward variables
    """
    # print("start calculate beta!")
    x = individual.matrix_sg.shape[1]  # initial the x-axis length(the type number in sg, pow(2,3)=8 normally)
    y = J   # initial y-axis length(the type number upperbound in hg)
    z = individual.matrix_sg.shape[0]   # initial z-axis length(total locus number)
    beta = test.np.zeros((x, y, z), dtype=test.np.float)
    individual.backward = [x-1 for x in individual.start_index]     # Sg的backward参数，计算branch用， start index -1
    for segment in reference.segments:  # Hg的backward参数
        reference.backward.append(segment.start_position-1)
    theta1 = (2 * K + theta) / (2 * K + 2 * theta)
    theta2 = 1 - theta1
    trans_occur = [2, 3, 4]
    trans_dict = None
    for m in range(z-1, -1, -1):
        if individual.heterozygous[m] is True:
            heterozygous = x / 2
        else:
            heterozygous = x
        if m == z-1:
            thou1 = thou[m-1]
            thou2 = 1-thou1
        else:
            thou1 = thou[m]
            thou2 = 1-thou1
        now_segment_idx = reference.seg_idx_dict.get(m)
        now_segment = reference.find_segment(now_segment_idx)
        now_dict = now_segment.haplotype_link
        if m < z-1:  # in this part, pre means m-1 position
            next_segment_idx = reference.seg_idx_dict.get(m+1)
            next_segment = reference.find_segment(next_segment_idx)
            next_dict = next_segment.haplotype_link
            m_position = m + 1 - next_segment.start_position
        else:   # The last node has no next node information
            next_segment_idx = None
            next_segment = None
            next_dict = None
            m_position = None
        branch = backward_index_branch(m, reference.backward, individual, Pivot)
        if branch in trans_occur:
            if m < Pivot-1:
                trans_dict = reference.find_transition(next_segment_idx - 1).transition
            elif m == Pivot-1:
                trans_dict = reference.find_transition(reference.transitions_count-1).transition
                trans_dict = trans_dict[0]
            else:
                '''测试用的是-1，真实情况是-2，记得回来改'''
                trans_dict = reference.find_transition(next_segment_idx - 2).transition
        if branch == -1:
            for i in range(x):
                for column in range(len(now_dict)):
                    beta[i][column][m] = 1
        elif branch == 1:
            for i in range(x):
                column = 0
                pre_sum = 0
                for j, values in now_dict.items():
                    if i == 0:
                        if column == 0:
                            temp_sum = 0
                            for u in range(x):
                                observed = individual.matrix_sg[m + 1][u]
                                v = 0
                                for temp_j, temp_values in next_dict.items():
                                    hidden = next_segment.matrix_form[m_position][v][0]
                                    temp_sum += beta[u][v][m+1] * inner_transition_prob(j, values, now_dict, temp_j, temp_values, next_dict, m, thou1, thou2, reference, K) \
                                                * emission_prob(observed, hidden, theta1, theta2, heterozygous)
                                    v += 1
                            pre_sum = temp_sum
                            beta[i][column][m] = temp_sum
                        else:
                            total_delta = 0
                            for r in range(x):
                                observed = individual.matrix_sg[m + 1][r]
                                hidden_i1 = next_segment.matrix_form[m_position][column][0]
                                hidden_i = next_segment.matrix_form[m_position][column-1][0]
                                this = beta[r][column][m + 1] * emission_prob(observed, hidden_i1, theta1, theta2, heterozygous) - beta[r][column - 1][m + 1] * emission_prob(observed, hidden_i, theta1, theta2, heterozygous)
                                total_delta += this
                            delta = total_delta * thou2
                            temp_sum = pre_sum + delta
                            pre_sum = temp_sum
                            beta[i][column][m] = temp_sum
                    else:
                        beta[i][column][m] = beta[0][column][m]
                    column += 1
        elif branch == 2:
            for i in range(x):
                observed = individual.matrix_sg[m + 1][i]
                column = 0
                for j, values in now_dict.items():
                    temp_sum = 0
                    v = 0
                    for temp_j, temp_values in next_dict.items():
                        hidden = next_segment.matrix_form[m_position][v][0]
                        temp_sum += beta[i][v][m+1] * inter_transition_prob(j, values, now_dict, temp_j, temp_values, next_dict, next_segment_idx, m, thou1, thou2, reference, K, Pivot, trans_dict) \
                                    * emission_prob(observed, hidden, theta1, theta2, heterozygous)
                        v += 1
                    beta[i][column][m] = temp_sum
                    column += 1
        elif branch in (3, 4):
            for i in range(x):
                column = 0
                for j, values in now_dict.items():
                    temp_sum = 0
                    for u in range(x):
                        observed = individual.matrix_sg[m + 1][u]
                        v = 0
                        for temp_j, temp_values in next_dict.items():
                            hidden = next_segment.matrix_form[m_position][v][0]
                            emi_prob = emission_prob(observed, hidden, theta1, theta2, heterozygous)
                            trans_prob = inter_transition_prob(j, values, now_dict, temp_j, temp_values, next_dict, next_segment_idx, m, thou1, thou2, reference, K, Pivot, trans_dict)
                            temp_sum += beta[u][v][m+1] * trans_prob * emi_prob
                            v += 1
                    beta[i][column][m] = temp_sum
                    column += 1
        else:
            for i in range(x):
                observed = individual.matrix_sg[m + 1][i]
                column = 0
                pre_sum = 0
                for j, values in now_dict.items():
                    if column == 0:
                        temp_sum = 0
                        v = 0
                        for temp_j, temp_values in next_dict.items():
                            hidden = next_segment.matrix_form[m_position][v][0]
                            temp_sum += beta[i][v][m+1] * inner_transition_prob(j, values, now_dict, temp_j, temp_values, next_dict, m, thou1, thou2, reference, K) * emission_prob(observed, hidden, theta1, theta2, heterozygous)
                            v += 1
                        pre_sum = temp_sum
                        beta[i][column][m] = temp_sum
                    else:
                        hidden_i1 = next_segment.matrix_form[m_position][column][0]
                        hidden_i = next_segment.matrix_form[m_position][column-1][0]
                        delta = (beta[i][column][m+1] * emission_prob(observed, hidden_i1, theta1, theta2, heterozygous) - beta[i][column-1][m+1] * emission_prob(observed, hidden_i, theta1, theta2, heterozygous)) * thou2
                        temp_sum = pre_sum + delta
                        pre_sum = temp_sum
                        beta[i][column][m] = temp_sum
                    column += 1
        normalize(m, beta)
    return beta


def xi_branch(m, hg_seg_end, Pivot):
    branch = 0
    if m in hg_seg_end:
        branch = 1
        if m == Pivot-1:
            branch = 2
    return branch


def cal_expectation(alpha, beta, reference, individual, thou, K, theta, J, Pivot):
    """
    求每一段的分布以及两段的联合分布
    :param alpha: forward prob
    :param beta: backward prob
    :param reference: Hg
    :param individual: Sg
    :param thou: recombination
    :param K: population
    :param theta: mutation
    :param J: condition states
    :param Pivot: pivot
    :return: Matirx: P(xi = up) and  Matrix: P(xi = up, x(i+1) = down)
    """
    states = alpha.shape[0]     # 每段candidate种类，|I| = pow(2, B)
    segment_expectations = test.np.zeros((len(individual.backward), states))    # P(Xm = hm| HMM), 矩阵存储
    transition_expectations = test.np.zeros((len(individual.backward)-1, states, states))   # P(Xm = hm,Xm+1 = hm+1|HMM)
    segment_count = 0   # 记录当前是哪一段
    transition_count = 0    # 记录当前是哪一个transition
    theta1 = (2 * K + theta) / (2 * K + 2 * theta)   # mutation参数
    theta2 = 1 - theta1
    trans_occur = [1, 2]
    for m in individual.backward:   # 对于Sg的每一个分段点。由于是m和(m+1)，因此用分段点上游节点作为index
        if individual.heterozygous[m] is True:
            heterozygous = states / 2
        else:
            heterozygous = states
        if m == -1:     # 初始概率，P(X1 = h1|HMM)
            dominator = beta[:, :, m+1]
            for i in range(states):
                numerator = beta[i, :, m+1]
                segment_expectations[segment_count][i] = test.np.sum(numerator) / test.np.sum(dominator)
            segment_count += 1
        else:   # 其余Hg分段
            thou1 = thou[m]
            thou2 = 1 - thou1
            now_segment = reference.find_segment(reference.find_segment_idx(m))     # Sg分段点上游段index
            now_dict = now_segment.haplotype_link
            next_segment_idx = reference.find_segment_idx(m+1)      # Sg分段点下游段index
            next_segment = reference.find_segment(next_segment_idx)
            next_dict = next_segment.haplotype_link
            branch = xi_branch(m, reference.backward, Pivot)    # 判断m是否处于Hg transition位置
            if branch in trans_occur:   # 如果是Hg transition则求transtion信息
                if m < Pivot-1:     # 求当前m对应的transition信息
                    trans_dict = reference.find_transition(next_segment_idx - 1).transition
                elif m == Pivot-1:
                    trans_dict = reference.find_transition(reference.total_transitions()-1).transition
                    trans_dict = trans_dict[0]
                else:
                    '''测试用的是-1，真实情况是-2，记得回来改'''
                    trans_dict = reference.find_transition(next_segment_idx - 2).transition
            else:  # 否则transition信息为None
                trans_dict = None
            m_1_position = m + 1 - next_segment.start_position
            if branch not in trans_occur:     # 如果只是Sg transition
                numerator_array = []
                dominator = 0
                for idx1 in range(states):  # m, hm
                    for idx2 in range(states):  # m+1, h(m+1)
                        column1 = 0
                        observed = individual.matrix_sg[m+1][idx2]
                        pre_central_sum = 0
                        f_j1 = []
                        for j1, now_value in now_dict.items():  # m, zm
                            temp_alpha = alpha[idx1][column1][m]
                            central_sum = 0
                            column2 = 0
                            if column1 == 0:
                                for j2, next_value in next_dict.items():    # m+1, z(m+1)
                                    hidden = next_segment.matrix_form[m_1_position][column2][0]
                                    emi_prob = emission_prob(observed, hidden, theta1, theta2, heterozygous)
                                    temp_beta = beta[idx2][column2][m+1]
                                    central_sum += emi_prob * temp_beta * inner_transition_prob(j1, now_value, now_dict, j2, next_value, next_dict, m, thou1, thou2, reference, K)
                                    column2 += 1
                                pre_central_sum = central_sum
                                f_j1.append(temp_alpha * central_sum)
                            else:
                                hidden_1 = next_segment.matrix_form[m_1_position][column1][0]
                                hidden = next_segment.matrix_form[m_1_position][column1-1][0]
                                emi_prob1 = emission_prob(observed, hidden_1, theta1, theta2, heterozygous)
                                emi_prob = emission_prob(observed, hidden, theta1, theta2, heterozygous)
                                delta = emi_prob1 * beta[idx2][column1][m+1] - emi_prob * beta[idx2][column1-1][m+1]
                                delta = delta * thou2
                                central_sum = pre_central_sum + delta
                                pre_central_sum = central_sum
                                f_j1.append(temp_alpha * central_sum)
                            column1 += 1
                        numerator = sum(f_j1)
                        dominator += numerator
                        numerator_array.append((idx1, idx2, numerator))
                        f_j1.clear()
                for t in numerator_array:
                    result = (t[0], t[1], t[2] / dominator)
                    transition_expectations[transition_count][result[0]][result[1]] = result[2]
            else:   # 如果同时是Hg transition和Sg transition
                numerator_array = []
                dominator = 0
                for idx1 in range(states):  # m position, hm
                    for idx2 in range(states):  # m+1 position, h(m+1)
                        temp_numerator = 0
                        column1 = 0
                        observed = individual.matrix_sg[m+1][idx2]
                        for j1, now_value in now_dict.items():
                            column2 = 0
                            temp_alpha = alpha[idx1][column1][m]
                            for j2, next_value in next_dict.items():
                                hidden = next_segment.matrix_form[m_1_position][column2][0]
                                emi_prob = emission_prob(observed, hidden, theta1, theta2, heterozygous)
                                trans_prob = inter_transition_prob(j1, now_value, now_dict, j2, next_value, next_dict, next_segment_idx, m, thou1, thou2, reference, K, Pivot, trans_dict)
                                temp_numerator += temp_alpha * trans_prob * emi_prob * beta[idx2][column2][m+1]
                                column2 += 1
                            column1 += 1
                        dominator += temp_numerator
                        numerator_array.append((idx1, idx2, temp_numerator))
                for t in numerator_array:
                    result = (t[0], t[1], t[2] / dominator)
                    transition_expectations[transition_count][result[0]][result[1]] = result[2]
            transition_count += 1
    for seg_id in range(1, segment_expectations.shape[0]-1):
        for possible in range(states):
            marginal_trans_expectation = transition_expectations[seg_id, possible, :]
            segment_expectations[seg_id][possible] = test.np.sum(marginal_trans_expectation)
    dip_segment_expectations = test.np.zeros(shape=(segment_expectations.shape[0], states))
    dip_transition_expectations = test.np.zeros(shape=(transition_expectations.shape[0], states, states))
    for seg_id in range(dip_segment_expectations.shape[0]-1):
        for possible in range(states):
            dip_segment_expectations[seg_id][possible] = segment_expectations[seg_id][possible] * segment_expectations[seg_id][states-1-possible]
    for trans_id in range(dip_transition_expectations.shape[0]):
        for up in range(states):
            for down in range(states):
                dip_transition_expectations[trans_id][up][down] = transition_expectations[trans_id][up][down] * transition_expectations[trans_id][states-1-up][states-1-down]
    return dip_segment_expectations, dip_transition_expectations


def cal_weight(segment_expectations, transition_expectations):
    """
    求权重。将每一段的分布求对数，将联合分布变成条件分布并求对数。同时取反，输出维特比算法的输入
    :param segment_expectations: P(Segment i = up)
    :param transition_expectations:  P(Segment i = up, Segment (i+1) = down)
    :return: log weight, in order to avoid underflow in viterbi algorithm
    """
    x = transition_expectations.shape[0]    # transition数
    y = transition_expectations.shape[1]    # 上游状态数
    z = transition_expectations.shape[2]    # 下游状态数
    segment_weight = test.np.zeros((x+1, y))    # 段数比分割数多1
    transition_weight = test.np.zeros((x, y, z))
    for n in range(x):
        for idx_up in range(y):
            marginalization = log10(segment_expectations[n][idx_up])
            for idx_down in range(z):
                transition_weight[n][idx_up][idx_down] = -(log10(transition_expectations[n][idx_up][idx_down]) - marginalization)
    for n in range(segment_expectations.shape[0]-1):
        for idx_type in range(segment_expectations.shape[1]):
            segment_weight[n][idx_type] = -(log10(segment_expectations[n][idx_type]) + log10(segment_expectations[n][y-1-idx_type]))
    return segment_weight, transition_weight


def viterbi(segment_weight, transition_weight): 
    """
    计算最优轨迹，argmaxP(X|HMM)
    :param segment_weight:初始segment权重
    :param transition_weight: 每一个transition权重
    :return: X_star
    """
    trajectory = []  # haplotype 1
    complementary_trajectory = []   # haplotype 2
    segment_number = segment_weight.shape[0]    # N = |Sg.cut|
    type_idx = segment_weight.shape[1]      # |I| = pow(2, B)
    viterbi_information = test.np.zeros((segment_number, type_idx), dtype=object)   # 维特比最优路径，1是权重和，2是上游
    for n in range(segment_number):
        for i in range(type_idx):
            if n == 0:
                viterbi_information[n][i] = (segment_weight[n][i], -1)
            else:
                temp_sum_weight = []
                for upstream in range(type_idx):
                    temp_sum_weight.append(viterbi_information[n-1][upstream][0] + transition_weight[n-1][upstream][i])
                opt_weight = min(temp_sum_weight)
                opt_parent = temp_sum_weight.index(min(temp_sum_weight))
                viterbi_information[n][i] = (opt_weight, opt_parent)
    terminate_information = viterbi_information[-1, :]
    self_idx = terminate_information.argmin()
    print(terminate_information[self_idx][0])
    father_idx = terminate_information[self_idx][1]
    trajectory.append(self_idx)
    for i in range(segment_number-2, -1, -1):
        self_idx = father_idx
        trajectory.append(self_idx)
        father_idx = viterbi_information[i][self_idx][1]
    trajectory.reverse()
    for i in trajectory:
        another = type_idx - 1 - i
        complementary_trajectory.append(another)
    print(trajectory)
    print(complementary_trajectory)
    return trajectory, complementary_trajectory


def output(haplo1, haplo2, individual):
    """
    输入维特比算法求得的segment信息，计算对应的基因haplotype信息
    :param haplo1: haplotype1的segment信息
    :param haplo2: haplotype2的segment信息
    :param individual: Sg
    :return: haplotype1和haplotype2
    """
    M = individual.matrix_sg.shape[0]
    haplotype1 = test.np.zeros((M, 1), dtype=int)
    haplotype2 = haplotype1.copy()
    segment_count = 0
    while segment_count in range(len(individual.start_index)):
        seg_start = individual.start_index[segment_count]
        if segment_count < len(individual.start_index) - 1:
            seg_end = individual.start_index[segment_count+1]
        else:
            seg_end = M
        idx_1 = haplo1[segment_count]
        idx_2 = haplo2[segment_count]
        for loci in range(seg_start, seg_end, 1):
            haplotype1[loci][0] = individual.matrix_sg[loci][idx_1]
            haplotype2[loci][0] = individual.matrix_sg[loci][idx_2]
        segment_count += 1

    return haplotype1, haplotype2


def HMM_parameter(test_r, test_h, test_s, theta, J, Pivot, B, first_input):
    if first_input is False:
        Pivot = randrange(1, test_h.shape[0], step=1)
    print(Pivot)
    test_h, test_s, Ne, test_r, hg, individual1, K, J, Pivot = test.CHMM(test_r, test_h, test_s, J, Pivot, B, first_input)
    thou = test.distance_parameter_thou(K, test_r, Ne)
    """
    parameter = mp.Queue()
    alpha_process = mp.Process(target=cal_alpha, args=(thou, hg, individual1, K, theta, J, Pivot))
    beta_process = mp.Process(target=cal_beta, args=(thou, hg, individual1, K, theta, J, Pivot))
    alpha_process.start()
    beta_process.start()
    alpha_process.join()
    beta_process.join()
    alpha = parameter.get()
    beta = parameter.get()
    """
    alpha = cal_alpha(thou, hg, individual1, K, theta, J, Pivot)
    beta = cal_beta(thou, hg, individual1, K, theta, J, Pivot)
    print('alpha & beta!')
    dip_seg_expectation, dip_trans_expectation = cal_expectation(alpha, beta, hg, individual1, thou, K, theta, J, Pivot)
    return dip_seg_expectation, dip_trans_expectation, test_h, test_s, test_r, individual1, hg


def main_iteration(test_r, test_h, test_s, theta, J, Pivot, B):
    h, s, Ne, r, hg, individual1, K, J, Pivot = test.CHMM(test_r, test_h, test_s, J, Pivot, B, first_input=False)
    thou = test.distance_parameter_thou(K, r, Ne)
    """
    parameter = mp.Queue()
    alpha_process = mp.Process(target=cal_alpha, args=(thou, hg, individual1, K, theta, J, Pivot, parameter))
    beta_process = mp.Process(target=cal_beta, args=(thou, hg, individual1, K, theta, J, Pivot, parameter))
    alpha_process.start()
    beta_process.start()
    alpha_process.join()
    beta_process.join()
    print('alpha & beta!')
    alpha = parameter.get()
    beta = parameter.get()
    """
    alpha = cal_alpha(thou, hg, individual1, K, theta, J, Pivot)
    beta = cal_beta(thou, hg, individual1, K, theta, J, Pivot)
    seg_expectation, trans_expectation = cal_expectation(alpha, beta, hg, individual1, thou, K, theta, J, Pivot)
    seg_weight, trans_weight = cal_weight(seg_expectation, trans_expectation)
    print("gamma!")
    haplo1, haplo2 = viterbi(seg_weight, trans_weight)
    haplotype1, haplotype2 = output(haplo1, haplo2, individual1)
    print("viterbi!")
    return haplotype1, haplotype2, h, s, r, individual1, hg


def main():
    t0 = test.time()
    test_r, test_h, test_s, J, Pivot, B = input.all_input()
    h, s, Ne, r, hg, individual1, K, J, Pivot = test.CHMM(test_r, test_h, test_s, J, Pivot, B, first_input=False)
    theta = pow(10, -8)
    print('Construct complete!')
    thou = test.distance_parameter_thou(K, r, Ne)
    t1 = test.time()
    print(t1 - t0)
    """
    parameter = mp.Queue()
    alpha_process = mp.Process(target=cal_alpha, args=(thou, hg, individual1, K, theta, J, Pivot, parameter))
    beta_process = mp.Process(target=cal_beta, args=(thou, hg, individual1, K, theta, J, Pivot, parameter))
    alpha_process.start()
    beta_process.start()
    alpha_process.join()
    beta_process.join()
    # alpha = cal_alpha(thou, hg, individual1, K, theta, J, Pivot)
    t2 = test.time()
    print(t2-t1)
    alpha = parameter.get()
    beta = parameter.get()
    # beta = cal_beta(thou, hg, individual1, K, theta, J, Pivot)
    t3 = test.time()
    print(t3 - t2)
    """
    alpha = cal_alpha(thou, hg, individual1, K, theta, J, Pivot)
    t2 = test.time()
    print(t2 - t1)
    beta = cal_beta(thou, hg, individual1, K, theta, J, Pivot)
    t3 = test.time()
    print(t3 - t2)
    seg_expectation, trans_expectation = cal_expectation(alpha, beta, hg, individual1, thou, K, theta, J, Pivot)
    print((seg_expectation >= 0).all())
    print((trans_expectation > 0).all())
    seg_weight, trans_weight = cal_weight(seg_expectation, trans_expectation)
    t4 = test.time()
    print(t4 - t3)
    haplo1, haplo2 = viterbi(seg_weight, trans_weight)
    print(test.time() - t4)
    output(haplo1, haplo2, individual1)


if __name__ == '__main__':
    main()
