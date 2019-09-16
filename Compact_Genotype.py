import numpy as np
import itertools
import sys

S = [0, 1, 2, 2, 1, 2, 1, 1]


class Genotype(object):
    """

    """
    def __init__(self, hetero_limit):
        """Initializing parameters"""
        self.hetero_limit = hetero_limit
        self.start_index = []

    def Sg_gen(self, S):
        """Generate Sg given S"""
        if not all(isinstance(s, int) for s in S):
            print("Type of input S must be list of intergers!")
            sys.exit(-1)

        if not S.count(1) > self.hetero_limit:
            self.Sg = self._Sg_seg_gen(S)

        self.Sg = []
        index_cut = self._cut_S(S)
        #print(index_cut)
        for i in range(len(index_cut)-1):
            interval = S[index_cut[i]: index_cut[i+1]]
            self.Sg.append(self._Sg_seg_gen(interval))

    def _cut_S(self, S):
        """Cut S given number of heterozygote in a segment"""
        index_cut = [0]
        hetero_count = 0
        for index, i in enumerate(S):
            hetero_count += i % 2
            if hetero_count == self.hetero_limit:
                hetero_count = 0
                index_cut.append(index+1)
        self.start_index = index_cut[:-1]
        if index_cut[-1] == len(S):
            return index_cut
        else:
            index_cut.append(len(S))
            return index_cut

    def _Sg_seg_gen(self, S_seg):
        """Generate Sg_seg given S_seg"""
        N = 2**self.hetero_limit
        hetero_ST = map(list, itertools.product([0, 1],
                                                repeat=S_seg.count(1)))
        hetero_S = (''.join([str(j) for j in i]) for i in
                    np.array(list(hetero_ST)).T)
        Sg_dict = {0: '0'*N, 2: '1'*N}

        return [next(hetero_S) if i == 1 else Sg_dict[i] for i in S_seg]


def main():
    individual1 = Genotype(2)
    individual1.Sg_gen(S)
    Sg = np.zeros((len(S), 2**individual1.hetero_limit), int)
    start = 0
    for segment in range(len(individual1.Sg)):
        for row in range(len(individual1.Sg[segment])):
            for column in range(len(individual1.Sg[segment][row])):
                Sg[row+start][column] = individual1.Sg[segment][row][column]
        start += len(individual1.Sg[segment])
    print('finish')


if __name__ == '__main__':
    main()
