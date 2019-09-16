import numpy as np
from time import time
from collections import Counter

h1 = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
              [1, 1, 1, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 1],
              [1, 1, 1, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1, 0, 1],
              [1, 1, 1, 0, 0, 1, 0, 1],
              [0, 0, 0, 1, 1, 0, 1, 0],
              [0, 0, 0, 1, 1, 0, 1, 0]])


class Graph:
    def __init__(self, segments_count, transitions_count):
        self.segments_count = segments_count
        self.transitions_count = transitions_count
        self.segments = []
        self.transitions = []

    def total_segments(self):
        return self.segments_count

    def total_transitions(self):
        return self.transitions_count

    def add_segment(self, segment):
        self.segments.append(segment)

    def output_segment(self):
        for i in self.segments:
            print('segment index: %s, haplotype: %s' % (i.find_index(), i.haplotype_link))

    def find_segment(self, index):
        return self.segments[index]

    def add_transition(self, transition):
        self.transitions.append(transition)

    def output_transition(self):
        for i in self.transitions:
            print('transition index: %s, transitions: %s' % (i.find_index(), i.transition))

    def find_transition(self, index):
        return self.transitions[index]


class Segment:
    """
     :param source: segment source (1 represents forward, 0 represents backward)
    """
    def __init__(self, segment_index, start_position, segment_len, source):
        self.start_position = start_position
        self.segment_index = segment_index
        self.segment_len = segment_len
        self.source = source
        self.haplotype_link = {}
        self.matrix_form = [[]]

    def add_haplotype(self, segment_info):
        self.haplotype_link = segment_info.copy()

    def find_index(self):
        return self.segment_index

    def be_matrix(self):
        self.matrix_form = [[[]for column in range(len(self.haplotype_link))] for row in range(self.segment_len)]
        column = 0
        for i, j in self.haplotype_link.items():
            if self.source == 1:
                h = ('{0:0' + str(self.segment_len) + 'b}').format(i)
                h = h[::-1]
                for row in range(len(h)):
                    cell = [int(h[row]), j]
                    self.matrix_form[row][column] = cell
            else:
                h = ('{0:0' + str(self.segment_len) + 'b}').format(i)
                for row in range(len(h)):
                    cell = [int(h[row]), j]
                    self.matrix_form[row][column] = cell
            column += 1


    def output(self):
        print('segment index: %s, start point: %s, segment length: %s' % (self.segment_index, self.start_position, self.segment_len))
        if self.source == 1:
            for i, j in self.haplotype_link.items():
                h = ('{0:0' + str(self.segment_len) + 'b}').format(i)
                print('haplotype link: %s, weight: %s' % (h[::-1], j))
        else:
            for i, j in self.haplotype_link.items():
                h = ('{0:0' + str(self.segment_len) + 'b}').format(i)
                print('haplotype link: %s, weight: %s' % (h, j))

    def source_output(self):
        if self.source == 0:
            print('segment source: backward')
        else:
            print('segment source: forward')


class Node:
    def __init__(self, value, row, column, weight, segment_idx):
        """
        Class node, to define all the node information which comes from Hg
        :param value: genotype value of the node
        :param row: row number
        :param column: column numbeer
        :param weight: weight number
        :param parent: upstream node
        :param child: downstream node
        :param segment_idx: belonging segment
        """
        self.value = value
        self.row = row
        self.column = column
        self.weight = weight
        self.parent = None
        self.child = None
        self.segment_idx = segment_idx

    def link_parent(self, father_node):
        self.parent = father_node

    def link_child(self, child_node):
        self.child = child_node

    def find_value(self):
        """

        :return: genotype value of the node
        """
        return self.value

    def find_weight(self):
        """

        :return: weight of the node
        """
        return self.weight

    def find_location(self):
        """

        :return: location in graph of the node
        """
        loc = [self.row, self.column]
        return loc

    def find_segment(self):
        """

        :return: segment index of the node
        """
        return self.segment_idx

    def find_parent(self):
        """

        :return: father node
        """
        return self.parent

    def find_child(self):
        """

        :return: child node
        """
        return self.child

    def output(self):
        """
        print node information
        :return: void
        """
        value = self.find_value()
        position = self.find_location()
        weight = self.find_weight()
        segment = self.find_segment()
        print('genotype is:%s' % value, ' position is:%s' % position, ' weight :%s' % weight, ' segment index:%s' % segment)


class Transition:
    """transition information"""
    def __init__(self, index, position, source):
        """

        :param index: transition index
        :param position: transition position (downstream segment start position)
        :param source: transition source (1 represents forward, 0 represents backward)
        """
        self.index = index
        self.position = position
        self.source = source
        self.transition = {}

    def find_index(self):
        return self.index

    def find_position(self):
        return self.position

    def add_transition(self, transition_info):
        self.transition = transition_info.copy()

    def cal_weight(self, upstream_type, downstream_type):
        return self.transition[(upstream_type, downstream_type)]

    def output(self):
        index = 0
        print('transition index: %s, transition position: %s' %(self.index, self.position))
        for i, j in self.transition.items():
            print('the %sth connection: %s, weight: %s ' % (index, i, j))
            index += 1

    def source_output(self):
        if self.source == 0:
            print('transition source: backward')
        else:
            print('transition source: forward')


def inverted_read_information(h, J, pivot):
    """
    this function is for backward reading
    :param h: primal graph
    :param J: type number limit
    :param pivot: initial start point
    :return: segment information, transition information, cut index information
    """
    if pivot == 0:
        return [], [], []
    curr_idx = pivot-1
    mark = curr_idx
    cut_index = [mark]
    segment_length = []
    int_values = []
    nb_segment = 0
    binary_dict = {}
    while curr_idx >= 0:
        # init the mark_th segement in int_values
        int_value = h[mark]
        this = int_value
        # mv current index to next()
        curr_idx -= 1
        while curr_idx >= 0:
            exp = mark - curr_idx
            if exp not in binary_dict.keys():
                binary_dict[exp] = 2**exp
            this = [(i + binary_dict[exp]) if j == 1 else i for i, j in zip(this, h[curr_idx])]
            if len(set(this)) <= J and curr_idx >= 0:
                int_value = this
                curr_idx -= 1
            else:
                break

        int_values.append(int_value)
        mark = curr_idx
        cut_index.append(mark)
        nb_segment += 1

    transitions = []
    for i in range(len(int_values) - 1):
        transitions.append(dict(Counter([(i, j) for i, j in zip(int_values[i], int_values[i + 1])])))
    cut_index = cut_index[:-1]
    return [dict(Counter(i)) for i in int_values], transitions, cut_index[::-1]


def read_information(h, J, pivot):
    """
    this function is for forward reading
    :param h: primal graph
    :param J: type number limit
    :param pivot: initial start point
    :return: segment information, transition information, cut index information
    """
    if pivot == h.shape[0]:
        return [], [], []
    curr_idx = pivot
    # mark represents starting point of each segment
    mark = curr_idx
    # list of mark for each segment
    cut_index = [mark]
    # list of list of int values of each segment
    segment_length = []
    int_values = []
    # incremental count of segments
    nb_segment = 0

    h_shape = h.shape[0]
    binary_dict = {}
    while curr_idx < h_shape:
        # init the mark_th segement in int_values
        int_value = h[mark]
        this = int_value
        # mv current index to next()
        curr_idx += 1
        while curr_idx < h_shape:
            exp = curr_idx-mark
            if not exp in binary_dict.keys():
                binary_dict[exp] = 2**exp
            this = [(i + binary_dict[exp]) if j == 1 else i for i, j in zip(this, h[curr_idx])]
            if len(set(this)) <= J and curr_idx < h_shape:
                int_value = this
                curr_idx += 1
            else:
                break

        int_values.append(int_value)
        mark = curr_idx
        cut_index.append(mark)
        nb_segment += 1

    transitions = []
    for i in range(len(int_values)-1):
        transitions.append(dict(Counter([(i, j) for i, j in zip(int_values[i], int_values[i+1])])))
    return [dict(Counter(i)) for i in int_values], transitions, cut_index[:-1]


def form_graph(r, c):
    h = np.random.randint(0, 2, size=(r, c))
    h = np.array(h)
    return h


def compact_graph(f_segment_info, f_transition_info, f_each_segment_length, f_cut_index\
                , b_segment_info, b_transition_info, b_each_segment_length, b_cut_index):
    """
    this function is for compact the primal graph according to all the information from previous function
    :param f_segment_info: forward segment information.
    Including:
    1. A dictionary whose keys represent the haplotype in this segment, whose values represent the weight of
    corresponding haplotype.
    2. Start point of each segment
    3. Length of each segment
    4. Segment index
    :param f_transition_info: forward transition information,
    Including:
    1. A dictionary whose key is a tuple where the first element is upstream haplotype, the second element is
    downstream haplotype,  whose value is the weight of corresponding transition
    :param f_each_segment_length: each segment length
    :param f_cut_index: the cut index. The most important property is at the first index of each segment
    Example, cut index = 2 means a cut between 1 and 2
    :param b_segment_info: similar
    :param b_transition_info: similar but the tuple is different, the first element is downstream, the second element
    is upstream
    :param b_each_segment_length:
    :param b_cut_index:
    :return: A compacted graph
    """
    hg = Graph(len(f_segment_info)+len(b_segment_info), len(f_transition_info)+len(b_transition_info))

    b_segment_info.reverse()
    b_current_segment_index = 0
    for s1 in b_segment_info:
        current_segment = Segment(b_current_segment_index,\
                                  b_cut_index[b_current_segment_index]+1-b_each_segment_length[b_current_segment_index]\
                                  , b_each_segment_length[b_current_segment_index], 0)
        current_segment.add_haplotype(b_segment_info[b_current_segment_index])
        hg.add_segment(current_segment)
        b_current_segment_index += 1

    f_current_segment_index = 0
    for s2 in f_segment_info:
        current_segment = Segment(f_current_segment_index+b_current_segment_index, f_cut_index[f_current_segment_index]\
                                  , f_each_segment_length[f_current_segment_index], 1)
        current_segment.add_haplotype(f_segment_info[f_current_segment_index])
        hg.add_segment(current_segment)
        f_current_segment_index += 1

    b_transition_info.reverse()
    b_current_transition_index = 0
    for t1 in b_transition_info:
        current_transition = Transition(b_current_transition_index, b_cut_index[b_current_transition_index]+1, 0)
        current_transition.add_transition(b_transition_info[b_current_transition_index])
        hg.add_transition(current_transition)
        b_current_transition_index += 1

    f_current_transition_index = 0
    for t2 in f_transition_info:
        current_transition = Transition(f_current_transition_index+b_current_transition_index,\
                                        f_cut_index[f_current_transition_index + 1], 1)
        current_transition.add_transition(f_transition_info[f_current_transition_index])
        hg.add_transition(current_transition)
        f_current_transition_index += 1

    return hg


def f_segment_length(terminal_position, cut_index):
    each_segment_length = []
    for i in range(1, len(cut_index)):
        each_segment_length.append(cut_index[i]-cut_index[i-1])
    each_segment_length.append(terminal_position-cut_index[-1])
    return each_segment_length


def b_segment_length(cut_index):
    each_segment_length = [cut_index[0]+1]
    for i in range(1, len(cut_index)):
        each_segment_length.append(cut_index[i]-cut_index[i-1])
    return each_segment_length


def compact_reference(h, J, Pivot):
    t0 = time()
    f_s_i, f_t_i, f_c_i = read_information(h1, J, Pivot)
    b_s_i, b_t_i, b_c_i = inverted_read_information(h1, J, Pivot)
    t1 = time()
    print(t1 - t0)

    f_e_s_l = f_segment_length(h.shape[0], f_c_i)
    b_e_s_l = b_segment_length(b_c_i)

    hg = compact_graph(f_s_i, f_t_i, f_e_s_l, f_c_i, b_s_i, b_t_i, b_e_s_l, b_c_i)
    print(time() - t1)
    print('finish')
    return hg


def main():

    h2 = form_graph(8500, 500)
    t0 = time()
    f_s_i, f_t_i, f_c_i = read_information(h1, 3, 0)
    b_s_i, b_t_i, b_c_i = inverted_read_information(h1, 3, 1)

    #print(f_s_i, f_t_i, f_c_i)
    #print(b_s_i, b_t_i, b_c_i)
    t1 = time()
    print(t1-t0)

    f_e_s_l = f_segment_length(h1.shape[0], f_c_i)
    b_e_s_l = b_segment_length(b_c_i)

    hg = compact_graph(f_s_i, f_t_i, f_e_s_l, f_c_i, b_s_i, b_t_i, b_e_s_l, b_c_i)
    print(time()-t1)
    print('finish')

    #hg = compact_reference(h1, 3, 4)
    for i1 in range(len(hg.segments)):
        hg.find_segment(i1).output()
        hg.find_segment(i1).source_output()
        for x in hg.find_segment(i1).haplotype_link.keys():
            print(x)
        hg.find_segment(i1).be_matrix()
    #hg.find_segment(1).output()
    for i2 in range(len(hg.transitions)):
        hg.find_transition(i2).output()
        hg.find_transition(i2).source_output()


if __name__ == '__main__':
    main()
