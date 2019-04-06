import pdb

# class Test:
#     class_var = 5

#     def __init__(self):
#         pass

# if __name__ == '__main__':
#     pdb.set_trace()
#     print('stop point')

class A:
    pass

class B:
    pass

class C:
    def __init__(self):
        pdb.set_trace()
        print('creating C')
        return

if __name__ == '__main__':
    c = C()
    print('stop')
