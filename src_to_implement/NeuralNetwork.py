import copy


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.current_label_tensor = label_tensor

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return self.loss_layer.forward(input_tensor, label_tensor)

    def backward(self):
        error_tensor = self.loss_layer.backward(self.current_label_tensor)

        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss_value = self.forward()
            self.loss.append(loss_value)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor


