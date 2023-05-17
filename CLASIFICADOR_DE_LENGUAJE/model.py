import nn

class LanguageClassificationModel(object):
    """
    Un modelo para la identificación de idiomas de una sola palabra.

    """
    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finlandés", "Holandés", "Polaco"]
        
        self.w = nn.Parameter(47, 100)
        self.w_h1 = nn.Parameter(100, 100)
        self.w_h2 = nn.Parameter(100, 100)
        self.w_f = nn.Parameter(100, 5)
    def run(self, xs):
        """
        Ejecuta el modelo para un lote de ejemplos.
        
        """
        
        def f(x, h):
            if not h:
                return nn.Linear(x, self.w)
            return nn.Linear(nn.ReLU(nn.Add(nn.Linear(x, self.w), nn.Linear(h, self.w_h1))), self.w_h2)

        h = None
        for x in xs:
            h = f(x, h)
        return nn.Linear(h, self.w_f)
    
    def get_loss(self, xs, y):
        """
        Calcula la pérdida de un lote de ejemplos.
        
        """
        return nn.SoftmaxLoss(self.run(xs), y)
    
    def train(self, dataset):
        """
        Entrenando el modelo
        
        """
        acc = 0
        alpha = -0.05
        count = 0
        while acc < 0.86:
            for xs, y in dataset.iterate_once(100):
                loss = self.get_loss(xs, y)
                grad_w, grad_w_h1, grad_w_h2, grad_w_f = nn.gradients(
                    loss, [self.w, self.w_h1, self.w_h2, self.w_f]
                    )

                self.w.update(grad_w, alpha)
                self.w_h1.update(grad_w_h1, alpha)
                self.w_h2.update(grad_w_h2, alpha)
                self.w_f.update(grad_w_f, alpha)
            count += 1
            
            acc = dataset.get_validation_accuracy()
            print(acc, alpha)


