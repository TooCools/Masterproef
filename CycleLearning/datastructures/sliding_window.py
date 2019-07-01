from collections import deque

from datastructures.data_structure import DataStructure

'''
A static sliding window data structure to battle conceptual drift
'''


class SlidingWindow(DataStructure):

    def __init__(self, size):
        '''
        Creates a static sliding window with a specific size
        :param size: size of static sliding window
        '''
        super().__init__(size)
        self.data = deque(maxlen=size)

    def add_element(self, el):
        self.data.append(el)
