from torch1 import Torch


class PyTorch(Torch):
    def __init__(self, name):
        super().__init__(name)
        self.pyname = 'py' + name
    
    def sub_print(self):
        print('pyname', self.pyname)
        print('name', self.name)
    
    def __str__(self):
        return f'pyname: {self.pyname}\nname: {self.name}'
    
    def __eq__(self, other):
        return self.pyname == other.pyname

def main():
    t1 = Torch('moon in woo')
    t1.print()
    t2 = PyTorch('moon in woo')
    t2.print()
    t2.sub_print()


if __name__ == '__main__':
    main()
