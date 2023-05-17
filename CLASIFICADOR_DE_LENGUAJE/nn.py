import numpy as np

def format_shape(shape):
    return "x".join(map(str, shape)) if shape else "()"

class Node(object):
    def __repr__(self):
        return "<{} shape={} at {}>".format(
            type(self).__name__, format_shape(self.data.shape), hex(id(self)))

class DataNode(Node):
    """
    Clase principal para los nodos
    
    """
    def __init__(self, data):
        self.parents = []
        self.data = data

    def _forward(self, *inputs):
        return self.data

    @staticmethod
    def _backward(gradient, *inputs):
        return []

class Parameter(DataNode):
    """
    Un nodo de parámetro almacena parámetros utilizados en una red neuronal (o perceptrón).
    """
    def __init__(self, *shape):
        assert len(shape) == 2, (
            "Shape must have 2 dimensions, instead has {}".format(len(shape)))
        assert all(isinstance(dim, int) and dim > 0 for dim in shape), (
            "Shape must consist of positive integers, got {!r}".format(shape))
        limit = np.sqrt(3.0 / np.mean(shape))
        data = np.random.uniform(low=-limit, high=limit, size=shape)
        super().__init__(data)

    def update(self, direction, multiplier):
        assert isinstance(direction, Constant), (
            "Update direction must be a {} node, instead has type {!r}".format(
                Constant.__name__, type(direction).__name__))
        assert direction.data.shape == self.data.shape, (
            "Update direction shape {} does not match parameter shape "
            "{}".format(
                format_shape(direction.data.shape),
                format_shape(self.data.shape)))
        assert isinstance(multiplier, (int, float)), (
            "Multiplier must be a Python scalar, instead has type {!r}".format(
                type(multiplier).__name__))
        self.data += multiplier * direction.data
        assert np.all(np.isfinite(self.data)), (
            "Parameter contains NaN or infinity after update, cannot continue")

class Constant(DataNode):
    """
    Un nodo constante para representar:
     * Funciones de entrada
     * Etiquetas de salida
     * Gradientes calculados por retropropagación

    """
    def __init__(self, data):
        assert isinstance(data, np.ndarray), (
            "Data should be a numpy array, instead has type {!r}".format(
                type(data).__name__))
        assert np.issubdtype(data.dtype, np.floating), (
            "Data should be a float array, instead has data type {!r}".format(
                data.dtype))
        super().__init__(data)

class FunctionNode(Node):
    def __init__(self, *parents):
        assert all(isinstance(parent, Node) for parent in parents), (
            "Inputs must be node objects, instead got types {!r}".format(
                tuple(type(parent).__name__ for parent in parents)))
        self.parents = parents
        self.data = self._forward(*(parent.data for parent in parents))

class Add(FunctionNode):
    """
    Agrega matrices por elementos.

     Entradas:
         x: un nodo con forma (batch_size x num_features)
         y: un Nodo con la misma forma que x
     Resultado:
         un nodo con forma (batch_size x num_features)
    """
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, "Expected 2 inputs, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "First input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        assert inputs[1].ndim == 2, (
            "Second input should have 2 dimensions, instead has {}".format(
                inputs[1].ndim))
        assert inputs[0].shape == inputs[1].shape, (
            "Input shapes should match, instead got {} and {}".format(
                format_shape(inputs[0].shape), format_shape(inputs[1].shape)))
        return inputs[0] + inputs[1]

    @staticmethod
    def _backward(gradient, *inputs):
        assert gradient.shape == inputs[0].shape
        return [gradient, gradient]

class AddBias(FunctionNode):
    """
    Agrega un vector de sesgo a cada vector de características
     Entradas:
         características: un nodo con forma (batch_size x num_features)
         bias: un Nodo con forma (1 x num_features)
     Producción:
         un nodo con forma (batch_size x num_features)
    """
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, "Expected 2 inputs, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "First input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        assert inputs[1].ndim == 2, (
            "Second input should have 2 dimensions, instead has {}".format(
                inputs[1].ndim))
        assert inputs[1].shape[0] == 1, (
            "First dimension of second input should be 1, instead got shape "
            "{}".format(format_shape(inputs[1].shape)))
        assert inputs[0].shape[1] == inputs[1].shape[1], (
            "Second dimension of inputs should match, instead got shapes {} "
            "and {}".format(
                format_shape(inputs[0].shape), format_shape(inputs[1].shape)))
        return inputs[0] + inputs[1]

    @staticmethod
    def _backward(gradient, *inputs):
        assert gradient.shape == inputs[0].shape
        return [gradient, np.sum(gradient, axis=0, keepdims=True)]

class DotProduct(FunctionNode):
    """
    Producto punto por lotes

     Entradas:
         características: un nodo con forma (batch_size x num_features)
         pesos: un Nodo con forma (1 x num_features)
     Salida: un nodo con forma (batch_size x 1)
    """
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, "Expected 2 inputs, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "First input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        assert inputs[1].ndim == 2, (
            "Second input should have 2 dimensions, instead has {}".format(
                inputs[1].ndim))
        assert inputs[1].shape[0] == 1, (
            "First dimension of second input should be 1, instead got shape "
            "{}".format(format_shape(inputs[1].shape)))
        assert inputs[0].shape[1] == inputs[1].shape[1], (
            "Second dimension of inputs should match, instead got shapes {} "
            "and {}".format(
                format_shape(inputs[0].shape), format_shape(inputs[1].shape)))
        return np.dot(inputs[0], inputs[1].T)

    @staticmethod
    def _backward(gradient, *inputs):
        raise NotImplementedError(
            "Backpropagation through DotProduct nodes is not needed in this "
            "assignment")

class Linear(FunctionNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, "Expected 2 inputs, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "First input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        assert inputs[1].ndim == 2, (
            "Second input should have 2 dimensions, instead has {}".format(
                inputs[1].ndim))
        assert inputs[0].shape[1] == inputs[1].shape[0], (
            "Second dimension of first input should match first dimension of "
            "second input, instead got shapes {} and {}".format(
                format_shape(inputs[0].shape), format_shape(inputs[1].shape)))
        return np.dot(inputs[0], inputs[1])

    @staticmethod
    def _backward(gradient, *inputs):
        assert gradient.shape[0] == inputs[0].shape[0]
        assert gradient.shape[1] == inputs[1].shape[1]
        return [np.dot(gradient, inputs[1].T), np.dot(inputs[0].T, gradient)]

class ReLU(FunctionNode):
    
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 1, "Expected 1 input, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "Input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        return np.maximum(inputs[0], 0)

    @staticmethod
    def _backward(gradient, *inputs):
        assert gradient.shape == inputs[0].shape
        return [gradient * np.where(inputs[0] > 0, 1.0, 0.0)]

class SquareLoss(FunctionNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, "Expected 2 inputs, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "First input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        assert inputs[1].ndim == 2, (
            "Second input should have 2 dimensions, instead has {}".format(
                inputs[1].ndim))
        assert inputs[0].shape == inputs[1].shape, (
            "Input shapes should match, instead got {} and {}".format(
                format_shape(inputs[0].shape), format_shape(inputs[1].shape)))
        return np.mean(np.square(inputs[0] - inputs[1]) / 2)

    @staticmethod
    def _backward(gradient, *inputs):
        assert np.asarray(gradient).ndim == 0
        return [
            gradient * (inputs[0] - inputs[1]) / inputs[0].size,
            gradient * (inputs[1] - inputs[0]) / inputs[0].size
        ]

class SoftmaxLoss(FunctionNode):
    @staticmethod
    def log_softmax(logits):
        log_probs = logits - np.max(logits, axis=1, keepdims=True)
        log_probs -= np.log(np.sum(np.exp(log_probs), axis=1, keepdims=True))
        return log_probs

    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, "Expected 2 inputs, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "First input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        assert inputs[1].ndim == 2, (
            "Second input should have 2 dimensions, instead has {}".format(
                inputs[1].ndim))
        assert inputs[0].shape == inputs[1].shape, (
            "Input shapes should match, instead got {} and {}".format(
                format_shape(inputs[0].shape), format_shape(inputs[1].shape)))
        assert np.all(inputs[1] >= 0), (
            "All entries in the labels input must be non-negative")
        assert np.allclose(np.sum(inputs[1], axis=1), 1), (
            "Labels input must sum to 1 along each row")
        log_probs = SoftmaxLoss.log_softmax(inputs[0])
        return np.mean(-np.sum(inputs[1] * log_probs, axis=1))

    @staticmethod
    def _backward(gradient, *inputs):
        assert np.asarray(gradient).ndim == 0
        log_probs = SoftmaxLoss.log_softmax(inputs[0])
        return [
            gradient * (np.exp(log_probs) - inputs[1]) / inputs[0].shape[0],
            gradient * -log_probs / inputs[0].shape[0]
        ]

def gradients(loss, parameters):

    assert isinstance(loss, (SquareLoss, SoftmaxLoss)), (
        "Loss must be a loss node, instead has type {!r}".format(
            type(loss).__name__))
    assert all(isinstance(parameter, Parameter) for parameter in parameters), (
        "Parameters must all have type {}, instead got types {!r}".format(
            Parameter.__name__,
            tuple(type(parameter).__name__ for parameter in parameters)))
    assert not hasattr(loss, "used"), (
        "Loss node has already been used for backpropagation, cannot reuse")

    loss.used = True

    nodes = set()
    tape = []

    def visit(node):
        if node not in nodes:
            for parent in node.parents:
                visit(parent)
            nodes.add(node)
            tape.append(node)

    visit(loss)
    nodes |= set(parameters)

    grads = {node: np.zeros_like(node.data) for node in nodes}
    grads[loss] = 1.0

    for node in reversed(tape):
        parent_grads = node._backward(
            grads[node], *(parent.data for parent in node.parents))
        for parent, parent_grad in zip(node.parents, parent_grads):
            grads[parent] += parent_grad

    return [Constant(grads[parameter]) for parameter in parameters]

def as_scalar(node):

    assert isinstance(node, Node), (
        "Input must be a node object, instead has type {!r}".format(
            type(node).__name__))
    assert node.data.size == 1, (
        "Node has shape {}, cannot convert to a scalar".format(
            format_shape(node.data.shape)))
    return np.asscalar(node.data)
