import random

from xsrc.datastructures.data_structure import DataStructure


# Aggarwal, C. On Biased Reservoir Sampling in the Presence of Stream Evolution
# A class that keeps items in a fixed length list, biased to more recent elements
class BiasedReservoirSampling(DataStructure):
    def __init__(self, size):
        super().__init__(size)

    def add_element(self, el):
        fullness = len(self.data) / self.size
        self.data.append(el)
        if random.random() <= fullness:
            to_delete = random.choice(self.data)
            self.data.remove(to_delete)
