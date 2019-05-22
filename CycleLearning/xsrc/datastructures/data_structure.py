import random


class DataStructure():
    def __init__(self, size=999999999):
        self.size = size
        self.data = []

    def add_element(self, el):
        self.data.append(el)

    def add_elements(self, elements):
        for el in elements:
            self.add_element(el)

    def get_elements(self, shuffle=True):
        if shuffle:
            random.shuffle(self.data)
            return self.data
        return self.data
