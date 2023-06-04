def add(x, y):
    return x+y

def sub(x, y):
    return x-y
def mul(x, y):
    return x*y

def operate_on_values(x, y, fun):
    return fun(x, y)

print(operate_on_values(5, 6, sub))