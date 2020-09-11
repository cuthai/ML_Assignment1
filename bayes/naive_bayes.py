import json
import numpy as np
import matplotlib.pyplot as plt


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

        # Tune Results
        self.tune_parameter_list = []
        self.tune_accuracy_list = []

    def fit(self, data_split_name='test', p=None, m=None):
        # If no parameters grab the object's current theta and alpha
        if not p:
            p = self.p
        if not m:
            m = self.m

        self.construct_frequency_tree(data_split_name, p=p, m=m)

        return self.classify(data_split_name)

    def construct_frequency_tree(self, data_split_name, p, m):
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
                            (m * p)) /
                           (class_normalizer + m),
                        1: (len(data.loc[(data[column_name] == 1) & (data['Class_4'] == class_name)]) +
                            (m * p)) /
                           (class_normalizer + m)
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

    def tune(self):
        """
        Tune function

        This function uses the fit function targeted at the tune data split. It loops through a predefined list of theta
            and alpha in order to determine the highest accuracy score.
        """
        # Theta and alpha ranges. Theta starts at .5 and goes up by .5 to 10. Alpha start at 2 and goes up by 1 to 6
        p_list = np.linspace(.0005, .003, 6)
        m_list = np.linspace(.5, 3, 5)

        # Variables for remembering the optimal theta and alpha
        max_accuracy = 0
        optimal_p = .001
        optimal_m = 1

        # Go through each theta and alpha (similar to a grid search)
        for p in p_list:
            for m in m_list:
                # Fit and retrieve the accuracy
                accuracy = self.fit(data_split_name='tune', p=p, m=m)

                # Check for most optimal theta and alpha
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    optimal_p = p
                    optimal_m = m

                # Let's append the result back to the object for visualization later
                self.tune_parameter_list.append(f'{p}, {m}')
                self.tune_accuracy_list.append(accuracy)

        # After the loops are done, we'll set the model's theta and alpha
        self.p = optimal_p
        self.m = optimal_m

    def visualize_tune(self):
        """
        Tune visualization function

        This function uses the results of the tune function to create a plot graph

        :return: matplotlib saved jpg in output folder
        """
        # Figure / axis set up
        fig, ax = plt.subplots()

        # We'll plot the list of params and their accuracy
        ax.plot(self.tune_parameter_list, self.tune_accuracy_list, 'o')

        # Title
        ax.set_title(rf'{self.data_name} Tune Results - Optimal: p {self.p}, p {self.m}')

        # X axis
        ax.set_xlabel(r'Parameters - Major Ticks: p, Minor Ticks: m')
        ax.set_xlim(5, 30)
        ax.set_xticks(np.linspace(0, 25, 6))
        ax.set_xticklabels(np.linspace(.0005, .003, 6), rotation=45, fontsize=6)
        ax.minorticks_on()

        # Y axis
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='y', which='minor', bottom=False)

        # Saving
        plt.savefig(f'output\\naive_bayes_{self.data_name}_tune.jpg')
