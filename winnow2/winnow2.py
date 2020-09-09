import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json


class Winnow2:
    """
    Class Winnow2 to handle classification of the a single class of the data

    This class implements a fit, predict, and tune function for the Winnow2 function. The fit additionally uses a
        promotion and demotion function to handle the outcome of a classification. There are additional functions to
        summarize the three main fit, predict, and tune functions.
    """
    def __init__(self, etl, theta=.5, alpha=2):
        """
        Init function. Takes an etl object

        :param etl: etl, etl object with information and data from the data set
        :param theta: float, theta is the threshold for classification, coefficient above theta denotes target class = 1
        :param alpha: int, alpha is the number used for demotion or promotion
        """
        # ETL attributes
        self.etl = etl
        self.data = self.etl.data
        self.data_name = self.etl.data_name
        self.data_split = self.etl.data_split

        # Winnow2 Params
        self.theta = theta
        self.alpha = alpha
        self.weights = None

        # Tune Results
        self.tune_parameter_list = []
        self.tune_accuracy_list = []

        # Train Results
        self.train_classification_coefficient_list = None
        self.train_prediction_list = None
        self.train_accuracy = None
        self.train_results = None

        # Test Results
        self.test_classification_coefficient_list = None
        self.test_prediction_list = None
        self.test_accuracy = None
        self.test_results = None

        # Overall Summary
        self.summary = {}

    def fit(self, data_split_name='train', theta=None, alpha=None):
        """
        Fit function

        This function fits the Winnow2 weights to the specified data split name (train, test, tune). It will also use
            any specified theta or alpha. If none are supplied then the object's theta and alpha are used. Theta and
            alpha can be set in the init function, or set after a tune function, or specified here

        :param data_split_name: str, train or tune. Specifies the data split name to use
        :param theta: float, theta is the threshold for classification, coefficient above theta denotes target class = 1
        :param alpha: int, alpha is the number used for demotion or promotion
        :return classification_coefficient_list: list, classification coefficient, the number to compare to theta
        :return prediction_list: list, class, the assigned class after comparison of coefficient to theta
        :return accuracy: float, accuracy of the fit: (TP + TF) / N
        """
        # If no parameters grab the object's current theta and alpha
        if not theta:
            theta = self.theta
        if not alpha:
            alpha = self.alpha

        # Set up of data, x, y, as well as initial weights
        data = self.data_split[data_split_name]
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        y_name = data.keys()[-1]
        self.weights = [1] * len(x.keys())

        # Accuracy variables
        true_positive = 0
        true_negative = 0

        # Return list variables
        classification_coefficient_list = []
        prediction_list = []

        # Actual fitting of weights, loop over each row of x
        for row_index, row in x.iterrows():
            # Classification coefficient calculated using x * weights to compare to theta
            classification_coefficient = 0

            # List to hold which weights will be promoted or demoted (if = 1)
            weights_to_change = []

            # Now to go through each column of the row
            for column_index in range(len(row)):
                # Add weight * x to the current coefficient
                classification_coefficient += row[column_index] * self.weights[column_index]

                # If there was something to multiply we'll add it to weights_to_change
                if row[column_index] == 1:
                    weights_to_change.append(column_index)

            # After row is finished, compare the coefficient to theta to assign to class
            if classification_coefficient > theta:
                prediction = 1

                # If we have a false positive, we need to trigger a demotion on the weights_to_change
                if prediction != y[row_index]:
                    self.demotion(weights_to_change, alpha)

                # Other we have a true positive so let's give that to accuracy
                else:
                    true_positive += 1

            # If under theta
            else:
                prediction = 0

                # If we have a false negative, we need to trigger a promotion on the weights_to_change
                if prediction != y[row_index]:
                    self.promotion(weights_to_change, alpha)

                # Other we have a true negative so let's give that to accuracy
                else:
                    true_negative += 1

            # Add coefficient and class prediction to the appropriate list for returning
            classification_coefficient_list.append(classification_coefficient)
            prediction_list.append(prediction)

        # After loop is done, calculate accuracy
        accuracy = (true_positive + true_negative) / len(x)

        # If we are on the train data set, we need to set the object train attributes
        if data_split_name == 'train':
            self.train_classification_coefficient_list = classification_coefficient_list
            self.train_prediction_list = prediction_list
            self.train_accuracy = accuracy

            # Let's also go back to the original untransformed data set and attach our predictions there
            train_result_df = pd.DataFrame(self.data, index=data.index)
            train_result_df[y_name] = prediction_list
            self.train_results = train_result_df

        return classification_coefficient_list, prediction_list, accuracy

    def demotion(self, weights_to_change, alpha):
        """
        Demotion function

        :param weights_to_change: list, which weights need to be demoted
        :param alpha: int, amount to divide the weight by
        """
        # Go through the index of each of the weights_to_change to update
        for weight in weights_to_change:
            # We'll update the object weights itself to give this back to the fit function to continue
            self.weights[weight] = self.weights[weight] / alpha

    def promotion(self, weights_to_change, alpha):
        """
        Promotion function

        :param weights_to_change: list, which weights need to be promoted
        :param alpha: int, amount to multiply the weight by
        """
        # Go through the index of each of the weights_to_change to update
        for weight in weights_to_change:
            # We'll update the object weights itself to give this back to the fit function to continue
            self.weights[weight] = self.weights[weight] * alpha

    def tune(self):
        """
        Tune function

        This function uses the fit function targeted at the tune data split. It loops through a predefined list of theta
            and alpha in order to determine the highest accuracy score.
        """
        # Theta and alpha ranges. Theta starts at .5 and goes up by .5 to 10. Alpha start at 2 and goes up by 1 to 6
        theta_list = np.linspace(.5, 10, 20)
        alpha_list = np.linspace(2, 6, 5)

        # Variables for remembering the optimal theta and alpha
        max_accuracy = 0
        optimal_theta = .5
        optimal_alpha = 2

        # Go through each theta and alpha (similar to a grid search)
        for theta in theta_list:
            for alpha in alpha_list:
                # Fit and retrieve the accuracy
                accuracy = self.fit(data_split_name='tune', theta=theta, alpha=alpha)[2]

                # Check for most optimal theta and alpha
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    optimal_theta = theta
                    optimal_alpha = alpha

                # Let's append the result back to the object for visualization later
                self.tune_parameter_list.append(f'{theta}, {alpha}')
                self.tune_accuracy_list.append(accuracy)

        # After the loops are done, we'll set the model's theta and alpha
        self.theta = optimal_theta
        self.alpha = optimal_alpha

    def predict(self):
        """
        Predict function

        This is a clone of the fit function without the promotion and demotion steps

        :return classification_coefficient_list: list, classification coefficient, the number to compare to theta
        :return prediction_list: list, class, the assigned class after comparison of coefficient to theta
        :return accuracy: float, accuracy of the fit: (TP + TF) / N
        """
        # Set up of data, x, y
        data = self.data_split['test']
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        y_name = data.keys()[-1]

        # Accuracy variables
        true_positive = 0
        true_negative = 0

        # Return list variables
        classification_coefficient_list = []
        prediction_list = []

        # Actual fitting of weights, loop over each row of x
        for row_index, row in x.iterrows():
            # Classification coefficient calculated using x * weights to compare to theta
            classification_coefficient = 0

            # Now to go through each column of the row
            for column_index in range(len(row)):
                # Add weight * x to the current coefficient
                classification_coefficient += row[column_index] * self.weights[column_index]

            # After row is finished, compare the coefficient to theta to assign to class
            if classification_coefficient > self.theta:
                prediction = 1

                # If we have a true positive let's give that to accuracy
                if prediction == y[row_index]:
                    true_positive += 1

            # If under theta
            else:
                prediction = 0

                # If we have a true negative let's give that to accuracy
                if prediction == y[row_index]:
                    true_negative += 1

            # Add coefficient and class prediction to the appropriate list for returning
            classification_coefficient_list.append(classification_coefficient)
            prediction_list.append(prediction)

        # After loop is done, calculate accuracy
        accuracy = (true_positive + true_negative) / len(x)

        # We need to set the object test attributes
        self.test_classification_coefficient_list = classification_coefficient_list
        self.test_prediction_list = prediction_list
        self.test_accuracy = accuracy

        # Let's also go back to the original untransformed data set and attach our predictions there
        test_result_df = pd.DataFrame(self.data, index=data.index)
        test_result_df[y_name] = prediction_list
        self.test_results = test_result_df

        return classification_coefficient_list, prediction_list, accuracy

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
        ax.set_title(rf'{self.data_name} Tune Results - Optimal: $\theta$ {self.theta}, $\alpha$ {self.alpha}')

        # X axis
        ax.set_xlabel(r'Parameters - Major Ticks: $\theta$, Minor Ticks: $\alpha$')
        ax.set_xlim(5, 100)
        ax.set_xticks(np.linspace(0, 95, 20))
        ax.set_xticklabels(np.linspace(.5, 10, 20), rotation=45, fontsize=6)
        ax.minorticks_on()

        # Y axis
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='y', which='minor', bottom=False)

        # Saving
        plt.savefig(f'output\\{self.data_name}_tune.jpg')

    def create_and_save_summary(self):
        """
        Function to create a summary

        Creates a JSON summary for this object and outputs a JSON document to the output folder

        :return: JSON to output folder
        """
        # Set up the summary dictionary with tune, train, and test results
        self.summary = {
            'name': self.data_name,
            'tune': {
                'theta': self.theta,
                'alpha': self.alpha,
            },
            'train': {
                'accuracy': self.train_accuracy
            },
            'test': {
                'accuracy': self.test_accuracy
            }
        }

        # Saving
        with open(f'output\\{self.data_name}_summary.json', 'w') as file:
            json.dump(self.summary, file)

    def save_csv_results(self):
        """
        Function to output a csv of the results

        This uses the split of the original data set as output

        :return: csv to output folder
        """
        # Train
        self.test_results.to_csv(f'output\\{self.data_name}_test_results.csv')

        # Test
        self.train_results.to_csv(f'output\\{self.data_name}_train_results.csv')
