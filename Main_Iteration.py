import test
import alpha_beta
import input
import sys
import random as rd


def save(haplotype1, haplotype2):
    with open(sys.path[0]+'/output/result_demo.txt', 'w') as f:
        f.write(' '.join([str(x[0]) for x in haplotype1])+'\n')
        f.write(' '.join([str(x[0]) for x in haplotype2]))
    f.close()


def main():
    test_r, test_h, test_s, J, Pivot, B = input.all_input()
    # print(all(i > 0 for i in test_r))
    M = test_h.shape[0]
    theta = pow(10, -8)
    iter_count = 20
    haplotype1 = test.np.zeros((M, 1), dtype=int)
    haplotype2 = haplotype1.copy()
    t0 = test.time()
    convergence = []
    for i in range(iter_count):
        print(i)
        haplotype1, haplotype2, test_h, test_s, test_r, individual1, reference = alpha_beta.main_iteration(test_r, test_h, test_s, theta, J, Pivot, B)
        # print(reference.total_segments())
        # print(test.np.linalg.matrix_rank(test_h))
        print(test_h.shape[1])
        test_h = test.np.concatenate((test_h, haplotype1, haplotype2), axis=1)
        convergence.append(haplotype1)
        Pivot = rd.randint(1, M-1)
        diff = 0
        if i > 0:
            for m in range(M):
                if convergence[i][m][0] != convergence[i-1][m][0]:
                    diff += 1
        print(diff)

        t1 = test.time()
        print(t1-t0)
        t0 = t1
    save(haplotype1, haplotype2)


if __name__ == '__main__':
    main()
