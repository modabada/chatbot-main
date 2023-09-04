import time


class Torch:
    def __init__(self, name):
        self.name = name
    
    def print(self):
        print('this is torch class\n이름은', self.name)

def main():
    print('this is my first pytorch programming')
    t1 = Torch('torch1')
    t1.print()
    print(t1.name)


if __name__ == '__main__':
    main()
