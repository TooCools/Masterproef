import random

from datastructures.data_structure import DataStructure

'''
A biased reservoir sampling data structure based on C. Aggarwal's paper:  On Biased Reservoir Sampling in the Presence
of Stream Evolution. A data structure that keeps items in a fixed length list, biased to more recent elements

'''


class BiasedReservoirSampling(DataStructure):
    def __init__(self, size):
        '''
        Creates a biased reservoir of a certain size
        :param size: size of reservoir
        '''
        super().__init__(size)

    def add_element(self, el):
        fullness = len(self.data) / self.size
        self.data.append(el)
        if random.random() <= fullness:
            to_delete = random.choice(self.data)
            self.data.remove(to_delete)
