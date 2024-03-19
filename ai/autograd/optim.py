class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad