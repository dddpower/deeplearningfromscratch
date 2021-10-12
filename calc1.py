import numpy as np


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def func_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def func_2(x):
    return x[0] ** 2 + x[1] ** 2


def func_temp1(x0):
    return x0 ** 2 + 4.0 ** 2


def func_temp2(x1):
    return 3.0 ** 2 + x1 ** 2

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t.reshape(1, t. size)
        y = y.reshape(1, y. size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y(np.arange(batch_size), t) + 1e-7)) / batch_size


def softmax(a):
    exp_a = np.exp(a)
    return exp_a / np.sum(exp_a)


init_x = np.array([-3.0, 4.0])
print(gradient_descent(func_2, init_x, lr=0.1))