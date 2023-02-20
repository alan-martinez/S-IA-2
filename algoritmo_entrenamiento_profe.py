import numpy as np

x = np.array([[0,0], [0,1], [1,0], [1,1]])
d = np.array([0,0,0,1])

class Perceptron():
    def __init__(self, n):
        self.w = np.random.random(n+1)
    def train_step(self, xs, ds):
        e = ds - self.y(xs)
        self.w = self.w + 0.4 * e * xs
        return e
    def y(self, xs):
        return np.dot(self.w, xs) >= 0

neurona = Perceptron(2)
band = True
ds = d

xs = np.hstack((np.ones((4,1)),x))

while band:
    band = False
    for i in range(4):
        error = neurona.train_step(xs[i],ds[i])
        if error != 0:
            band = True

neurona.y(xs.T)
print(neurona.y(xs.T))
# print(neurona.w)