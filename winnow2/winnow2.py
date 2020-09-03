class Winnow2:
    def __init__(self, etl, theta=.5, alpha=.5):
        self.etl = etl
        self.data = self.etl.data
        self.data_name = self.etl.data_name

        self.threshold = theta
        self.alpha = alpha

    def classify(self):
        pass
