from collections import deque

from xsrc.datastructures.data_structure import DataStructure


class SlidingWindow(DataStructure):
    def __init__(self,size):
        super().__init__(size)
        self.data=deque(size)

    def add_element(self,el):
        self.data.append(el)
