import json


class NaiveBayes:
    def __init__(self, etl, p=.001, m=1):
        """
        Init function. Takes an etl object

        :param etl: etl, etl object with information and data from the data set
        """
        # ETL attributes
        self.etl = etl
        self.data = self.etl.data
        self.data_name = self.etl.data_name
        self.data_split = self.etl.data_split
        self.classes = self.etl.classes

        # Naive Bayes Params
        self.p = p
        self.m = m
        self.variable_frequency = {}

    def fit(self, data_split_name='test'):
        self.construct_frequency_tree(data_split_name)

        print(self.classify(data_split_name))

    def construct_frequency_tree(self, data_split_name):
        data = self.data_split[data_split_name]
        overall_normalizer = len(data)

        for class_name in [0, 1]:
            class_normalizer = len(data.loc[data['Class_4'] == class_name])
            frequency_dict = {
                'class_normalizer': class_normalizer,
                'class_frequency': class_normalizer / overall_normalizer
            }

            for column_name in data.keys()[:-self.classes]:
                frequency_dict.update({
                    column_name: {
                        0: (len(data.loc[(data[column_name] == 0) & (data['Class_4'] == class_name)]) +
                            (self.m * self.p)) /
                           (class_normalizer + self.m),
                        1: (len(data.loc[(data[column_name] == 1) & (data['Class_4'] == class_name)]) +
                            (self.m * self.p)) /
                           (class_normalizer + self.m)
                    }
                })

            self.variable_frequency.update({class_name: frequency_dict})

        x = json.dumps(self.variable_frequency)
        pass

    def classify(self, data_split_name):
        data = self.data_split[data_split_name]
        classification_list = []

        true_positive = 0
        true_negative = 0

        for index, row in data.iterrows():
            class_coefficient = [1] * self.classes

            for column_name in data.keys()[:-self.classes]:
                class_coefficient[0] = class_coefficient[0] * self.variable_frequency[0][column_name][row[column_name]]
                class_coefficient[1] = class_coefficient[1] * self.variable_frequency[1][column_name][row[column_name]]

            classification = class_coefficient.index(max(class_coefficient))
            classification_list.append(classification)

            if classification == row['Class_4'] and classification == 1:
                true_positive += 1
            elif classification == row['Class_4'] and classification == 0:
                true_negative += 1

        accuracy = (true_positive + true_negative) / len(data)

        return accuracy
