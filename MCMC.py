import alpha_beta
import input
import random as rd
import numpy as np
import test
import Main_Iteration
from math import log10
from math import log
from math import e


def init_sampling(sample_result, N, segment_expectation, transition_expectation, domain):
    """
    This function is used for sampling the initial point by Markovian property
    :param sample_result: a matrix that store the result
    :param N: segment number
    :param segment_expectation: HMM parameter
    :param transition_expectation: HMM parameter
    :param domain: domain of sample
    :return: an upgraded sample result matrix
    """
    for n in range(N):
        if n == 0:
            distribution = segment_expectation[0, :]
            dice_result = rd.choices(domain, distribution, k=1)
            sample_result[n][0] = int(dice_result[0])
        else:
            pre = sample_result[n-1][0]
            joint = transition_expectation[n-1, pre, :]
            distribution = joint / segment_expectation[n-1][pre]
            dice_result = rd.choices(domain, distribution, k=1)
            sample_result[n][0] = int(dice_result[0])


def iterate_sampling(sample_result, N, i, segment_expectation, transition_expectation, domain):
    """
    This function is applied for Gibbs sampling iteratively
    :param sample_result: a matrix which store the sample result
    :param N: segment number
    :param i: iteration index
    :param segment_expectation: HMM parameter
    :param transition_expectation: HMM parameter
    :param domain: domain of sample
    :return: an upgraded sample result matrix
    """
    for n in range(N):
        if n == 0:
            down_idx = sample_result[n+1][i-1]
            joint = transition_expectation[n, :, down_idx]
            distribution = joint / segment_expectation[n+1, down_idx]
            dice_result = rd.choices(domain, distribution, k=1)
            sample_result[n][i] = int(dice_result[0])
        elif n == N-1:
            up_idx = sample_result[n-1][i]
            joint = transition_expectation[n-1, up_idx, :]
            distribution = joint / segment_expectation[n-1, up_idx]
            dice_result = rd.choices(domain, distribution, k=1)
            sample_result[n][i] = int(dice_result[0])
        else:
            down_idx = sample_result[n+1][i-1]
            up_idx = sample_result[n-1][i]
            item = []
            for k in range(segment_expectation.shape[1]):
                then = transition_expectation[n, k, down_idx]
                pre = transition_expectation[n-1, up_idx, k]
                now = segment_expectation[n, k]
                item.append(then * pre / now)
            numerator = sum(item)
            distribution = [x/numerator for x in item]
            dice_result = rd.choices(domain, distribution, k=1)
            sample_result[n][i] = int(dice_result[0])


def burn_in_iter(burning_count, segment_expectation, transition_expectation, N, sample_result, domain, test_h, test_s, test_r, J, Pivot, B, theta, individual, first_input):
    """
    Burning iteration
    :param burning_count: burning iteration count
    :param segment_expectation: HMM parameter
    :param transition_expectation: HMM parameter
    :param N: segment number
    :param sample_result: a matrix which store the sample result
    :param domain: domain of sampling
    :return: complete burning iteration sample result
    """
    for i in range(burning_count):
        print("%d/%d Burn-In iteration" % (i+1, burning_count))
        if i == 0:
            init_sampling(sample_result, N, segment_expectation, transition_expectation, domain)
            haplo1 = [x for x in sample_result[:, 0]]
            haplo2 = [segment_expectation.shape[1] - 1 - x for x in haplo1]
            haplotype1, haplotype2 = alpha_beta.output(haplo1, haplo2, individual)
            test_h = np.concatenate((test_h, haplotype1, haplotype2), axis=1)
            segment_expectation, transition_expectation, test_h, test_s, test_r, individual, hg = alpha_beta.HMM_parameter(test_r, test_h, test_s, theta, J, Pivot, B, first_input)
        else:
            init_sampling(sample_result, N, segment_expectation, transition_expectation, domain)
            haplo1 = [x for x in sample_result[:, 0]]
            haplo2 = [segment_expectation.shape[1] - 1 - x for x in haplo1]
            haplotype1, haplotype2 = alpha_beta.output(haplo1, haplo2, individual)
            test_h = np.delete(test_h, [test_h.shape[1]-2, test_h.shape[1]-1], axis=1)
            test_h = np.concatenate((test_h, haplotype1, haplotype2), axis=1)
            segment_expectation, transition_expectation, test_h, test_s, test_r, individual, hg = alpha_beta.HMM_parameter(test_r, test_h, test_s, theta, J, Pivot, B, first_input)
    return segment_expectation, transition_expectation, test_h


def main_iter(main_count, segment_expectation, transition_expectation, N, sample_result, domain, test_h, test_s, test_r, J, Pivot, B, theta, individual, main_repeat, first_input):
    """
    Main iteration
    :param main_count: main iteration number
    :param segment_expectation: HMM parameter
    :param transition_expectation: HMM parameter
    :param N: segment number
    :param sample_result: a matrix which store sample result
    :param domain: domain of sampling
    :param test_h: Hg
    :param test_s: Sg
    :param test_r: r
    :param J: compact parameter
    :param Pivot: compact pivot
    :param B: segment length
    :param theta: mutation parameter
    :param individual: Sg
    :param main_repeat: sampling number in each iteration
    :return: complete main iteration sample result
    """
    segment_estimation = np.zeros(shape=(main_count, 1), dtype=object)
    transition_estimation = np.zeros(shape=(main_count, 1), dtype=object)
    for i in range(main_count):
        print("%d/%d Main iteration" % (i+1, main_count))
        segment_estimation[i][0] = segment_expectation
        transition_estimation[i][0] = transition_expectation
        init_sampling(sample_result, N, segment_expectation, transition_expectation, domain)
        haplo1 = [x for x in sample_result[:, 0]]
        haplo2 = [segment_expectation.shape[1]-1-x for x in haplo1]
        haplotype1, haplotype2 = alpha_beta.output(haplo1, haplo2, individual)
        test_h = np.delete(test_h, [test_h.shape[1] - 2, test_h.shape[1] - 1], axis=1)
        test_h = np.concatenate((test_h, haplotype1, haplotype2), axis=1)
        segment_expectation, transition_expectation, test_h, test_s, test_r, individual, hg = alpha_beta.HMM_parameter(test_r, test_h, test_s, theta, J, Pivot, B, first_input)
    return segment_estimation, transition_estimation


def posterior_estimation(segment_estimation, transition_estimation):

    """
    :param segment_estimation: segment probability of main result
    :param transition_estimation: transition probability of main result
    :return: initial likelihood estimation and conditional likelihood estimation
    """
    Nm = segment_estimation.shape[0]
    segment_number = segment_estimation[0][0].shape[0]
    states = segment_estimation[0][0].shape[1]
    segment_posterior = np.zeros(shape=(segment_number, states))
    transition_posterior = np.zeros(shape=(segment_number-1, states, states))
    temp = []
    for seg_id in range(segment_number):
        for possible in range(states):
            for i in range(Nm):
                temp.append(segment_estimation[i][0][seg_id][possible])
            segment_posterior[seg_id][possible] = sum(temp) / Nm
            temp.clear()
    temp.clear()
    for trans_id in range(segment_number-1):
        for up in range(states):
            for down in range(states):
                for i in range(Nm):
                    temp.append(transition_estimation[i][0][trans_id][up][down])
                transition_posterior[trans_id][up][down] = sum(temp) / Nm
                temp.clear()
    for seg_id in range(segment_number-1):
        this = segment_posterior[seg_id, :]
        norm = np.sum(this)
        for possible in range(states):
            segment_posterior[seg_id][possible] = segment_posterior[seg_id][possible] / norm
    for trans_id in range(segment_number-1):
        this = transition_posterior[trans_id, :, :]
        norm = np.sum(this)
        for up in range(states):
            for down in range(states):
                transition_posterior[trans_id][up][down] = transition_posterior[trans_id][up][down] / norm
    """
    Nm = main_result.shape[1]
    status = segment_expectation.shape[1]
    segment_estimation = np.zeros(shape=(segment_expectation.shape[0], segment_expectation.shape[1]))
    transition_estimation = np.zeros(shape=(segment_expectation.shape[0]-1, status, status))
    init = main_result[0, :]
    statistics = collections.Counter(init)
    for key, value in statistics.items():
        segment_estimation[0][key] = value / Nm
    for n in range(1, main_result.shape[0]):
        joint_result = main_result[n-1:n+1, :]
        joint_count = np.zeros(shape=(status, status))
        marginal_result = main_result[n-1, :]
        marginal_count = np.zeros(shape=(status, 1))
        for index in range(Nm):
            pre = marginal_result[index]
            marginal_count[pre] += 1
            now = joint_result[1][index]
            joint_count[pre][now] += 1
        for i in range(status):
            for j in range(status):
                if marginal_count[i] == 0:
                    transition_estimation[n-1][i][j] = 0
                else:
                    transition_estimation[n-1][i][j] = joint_count[i][j] / marginal_count[i]
    
    # ___ transition_estimation ___ #
    # row: upstream segment         #
    # column: downstream segment    #
    """
    return segment_posterior, transition_posterior


def cal_weight(segment_estimations, transition_estimations):
    """
    求权重。将每一段的分布求对数，同时取反，输出维特比算法的输入
    :param segment_estimations: P(Segment i = up)
    :param transition_estimations:  P(Segment i = up, Segment (i+1) = down)
    :return: log weight, in order to avoid underflow in viterbi algorithm
    """
    x = transition_estimations.shape[0]    # transition数
    y = transition_estimations.shape[1]    # 上游状态数
    z = transition_estimations.shape[2]    # 下游状态数
    segment_weight = test.np.zeros((x+1, y))    # 段数比分割数多1
    transition_weight = test.np.zeros((x, y, z))
    for seg_id in range(x+1):
        # norm = np.sum(segment_estimations[seg_id, :])
        for possible in range(y):
            segment_estimations[seg_id][possible] = segment_estimations[seg_id][possible]  # / norm
    for trans_id in range(x):
        # norm = np.sum(transition_estimations[trans_id, :, :])
        for up in range(y):
            for down in range(z):
                transition_estimations[trans_id][up][down] = transition_estimations[trans_id][up][down]  # / norm
    for idx_up in range(y):
        initial = log10(segment_estimations[0][idx_up])
        segment_weight[0][idx_up] = - initial
    for n in range(x):
        for idx_up in range(y):
            marginalization = log10(segment_estimations[n][idx_up])
            for idx_down in range(z):
                transition_weight[n][idx_up][idx_down] = -(log10(transition_estimations[n][idx_up][idx_down]) - marginalization)
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


def Gibbs(segment_expectation, transition_expectation, burning_count, main_count, test_h, test_s, test_r, J, Pivot, B, theta, individual1, main_repeat, first_input):
    burning_result = np.zeros(shape=(segment_expectation.shape[0], 1), dtype=int)
    main_result = np.zeros(shape=(segment_expectation.shape[0], 1), dtype=int)  # (main_count-1) * main_repeat+1)
    domain = range(segment_expectation.shape[1])
    N = segment_expectation.shape[0]
    segment_expectation, transition_expectation, test_h = burn_in_iter(burning_count, segment_expectation, transition_expectation, N, burning_result, domain, test_h, test_s, test_r, J, Pivot, B, theta, individual1, first_input)
    # main_result[:, 0] = burning_result[:, -1]
    segment_est, transition_est = main_iter(main_count, segment_expectation, transition_expectation, N, main_result, domain, test_h, test_s, test_r, J, Pivot, B, theta, individual1, main_repeat, first_input)
    print("Solving result!")
    segment_posterior, transition_posterior = posterior_estimation(segment_est, transition_est)
    segment_weight, transition_weight = cal_weight(segment_posterior, transition_posterior)
    trajectory1, trajectory2 = viterbi(segment_weight, transition_weight)
    phasing1, phasing2 = alpha_beta.output(trajectory1, trajectory2, individual1)
    return phasing1, phasing2


def main():
    test_r, test_h, test_s, J, Pivot, B = input.all_input()     # primal data
    theta = 1 / log(test_h.shape[1]-1, e)  # mutation parameter
    first_input = True
    dip_segment_expectation, dip_transition_expectation, test_h, test_s, test_r, individual1, hg = alpha_beta.HMM_parameter(test_r, test_h, test_s, theta, J, Pivot, B, first_input)  # compute HMM
    first_input = False
    burning_count = 5
    main_count = 15
    main_repeat = 1
    # ___ Gibbs sampling ___ #
    phasing1, phasing2 = Gibbs(dip_segment_expectation, dip_transition_expectation, burning_count, main_count, test_h, test_s, test_r, J, Pivot, B, theta, individual1, main_repeat, first_input)
    Main_Iteration.save(phasing1, phasing2)


if __name__ == '__main__':
    main()
