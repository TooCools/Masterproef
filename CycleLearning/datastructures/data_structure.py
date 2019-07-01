import random

'''
Base data structure. An unlimited size array that doesn't forget data
'''


class DataStructure():

    def __init__(self, size=999999999):
        self.size = size
        self.data = []

    def add_element(self, el):
        '''
        Adds el to the data structure
        :param el: element to add
        :return:
        '''
        self.data.append(el)

    def add_elements(self, elements):
        '''
        Adds an array of elements one by one to the data structure
        :param elements: array of elements to add
        :return:
        '''
        for el in elements:
            self.add_element(el)

    def get_elements(self, shuffle=True):
        '''
        Returns all elements in the data structure, shuffeled at random or in order
        :param shuffle: To shuffle or not to shuffle
        :return: elements in data structure
        '''
        if shuffle:
            random.shuffle(self.data)
            return self.data
        return self.data
