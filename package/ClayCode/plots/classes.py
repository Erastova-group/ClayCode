from __future__ import annotations


class Cutoff(str):
    def __new__(cls, length):
        string = f"{int(length):02}"
        return super().__new__(cls, string)

    def __init__(self, length):
        self.num = float(length)

    def __float__(self):
        return float(self.num)

    def __int__(self):
        return int(self.num)

    def __str__(self):
        return self


class Bins(str):
    def __new__(cls, length):
        string = f"{float(length):.02f}"[2:]
        return super().__new__(cls, string)

    def __init__(self, length):
        self.num = float(length)

    def __float__(self):
        return float(self.num)

    def __int__(self):
        return int(self.num)

    def __str__(self):
        return self
