import numpy as np


class Winnow2:
    def __init__(self, etl, theta=.5, alpha=2):
        self.etl = etl
        self.data = self.etl.data
        self.data_name = self.etl.data_name
        self.data_split = self.etl.data_split

        self.theta = theta
        self.alpha = alpha
        self.weights = None

        self.train_classification_coefficient_list = None
        self.train_prediction_list = None
        self.train_accuracy = None

        self.test_classification_coefficient_list = None
        self.test_prediction_list = None
        self.test_accuracy = None

    def fit(self, data_set='train', theta=None, alpha=None):
        if not theta:
            theta = self.theta
        if not alpha:
            alpha = self.alpha

        data = self.data_split[data_set]
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        self.weights = [1] * len(x.keys())

        true_positive = 0
        true_negative = 0

        classification_coefficient_list = []
        prediction_list = []

        for row_index, row in x.iterrows():
            weights_to_change = []
            classification_coefficient = 0

            for column_index in range(len(row)):
                classification_coefficient += row[column_index] * self.weights[column_index]
                if row[column_index] == 1:
                    weights_to_change.append(column_index)

            if classification_coefficient > theta:
                prediction = 1

                if prediction != y[row_index]:
                    self.demotion(weights_to_change, alpha)

                else:
                    true_positive += 1

            else:
                prediction = 0

                if prediction != y[row_index]:
                    self.promotion(weights_to_change, alpha)

                else:
                    true_negative += 1

            classification_coefficient_list.append(classification_coefficient)
            prediction_list.append(prediction)

        accuracy = (true_positive + true_negative) / len(x)

        if data_set == 'train':
            self.train_classification_coefficient_list = classification_coefficient_list
            self.train_prediction_list = prediction_list
            self.train_accuracy = accuracy

        return accuracy

    def demotion(self, weights_to_change, alpha):
        for weight in weights_to_change:
            self.weights[weight] = self.weights[weight] / alpha

    def promotion(self, weights_to_change, alpha):
        for weight in weights_to_change:
            self.weights[weight] = self.weights[weight] * alpha

    def tune(self):
        theta_list = np.linspace(.5, 10, 20)
        alpha_list = np.linspace(2, 6, 5)

        max_accuracy = 0
        optimal_theta = .5
        optimal_alpha = 2

        for theta in theta_list:
            for alpha in alpha_list:
                accuracy = self.fit(data_set='tune', theta=theta, alpha=alpha)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    optimal_theta = theta
                    optimal_alpha = alpha

        self.theta = optimal_theta
        self.alpha = optimal_alpha

    def predict(self):
        data = self.data_split['test']
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        true_positive = 0
        true_negative = 0

        classification_coefficient_list = []
        prediction_list = []

        for row_index, row in x.iterrows():
            classification_coefficient = 0

            for column_index in range(len(row)):
                classification_coefficient += row[column_index] * self.weights[column_index]

            if classification_coefficient > self.theta:
                prediction = 1

                if prediction == y[row_index]:
                    true_positive += 1

            else:
                prediction = 0

                if prediction == y[row_index]:
                    true_negative += 1

            classification_coefficient_list.append(classification_coefficient)
            prediction_list.append(prediction)

        accuracy = (true_positive + true_negative) / len(x)

        self.test_classification_coefficient_list = classification_coefficient_list
        self.test_prediction_list = prediction_list
        self.test_accuracy = accuracy

        return accuracy
