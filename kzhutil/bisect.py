from typing import Callable, Sequence


class MappingSpace:
    def __init__(self, arr: Sequence, key: Callable):
        self.arr = arr
        self.key = key

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return self.key(self.arr[i])
