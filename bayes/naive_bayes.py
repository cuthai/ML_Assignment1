import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NaiveBayes:
    """
    Class NaiveBayes to handle classification of data set using Naive Bayes method

    This class implements a fit, predict, and tune function for the NaiveBayes function. The fit additionally uses a
        construct variable_frequency and classification function to handle the outcome of a classification. There are
        additional functions to summarize the three main fit, predict, and tune functions.
    """
    def __init__(self, etl, p=.001, m=1):
        """
        Init function. Takes an etl object

        :param etl: etl, etl object with information and data from the data set
        :param p: float, value for smoothing function
        :param m: float, value for smoothing function
        """
        # ETL attributes
        self.etl = etl
        self.data = self.etl.data
        self.data_name = self.etl.data_name
        self.data_split = self.etl.data_split
        self.classes = self.etl.classes
        self.class_names = self.data.Class.unique().tolist()

        # After attribute set, we need to update the data split
        self.update_data_split()

        # Naive Bayes Params
        self.p = p
        self.m = m
        self.variable_frequency = {}

        # Tune Results
        self.tune_parameter_list = []
        self.tune_accuracy_list = []

        # Train Results
        self.train_accuracy = None
        self.train_results = None

        # Test Results
        self.test_accuracy = None
        self.test_results = None

        # Overall Summary
        self.summary = {}

    def update_data_split(self):
        """
        Function to update data split

        The Winnow2 needs a dummied class variable, naive bayes can instead use the native class function with class
            names. This function helps to revert from the dummied class variable back to the naive bayes for all of the
            data split
        """
        # Determine a class count to remove the number of dummied class variables
        if self.classes == 2:
            class_count = 1
        else:
            class_count = self.classes

        # Go through the train, test, tune and remove the old dummies and add the original class variable
        for key in self.data_split.keys():
            self.data_split[key] = self.data_split[key].iloc[:, :-class_count]
            self.data_split[key]['Class'] = self.etl.class_data_split[key]

    def construct_frequency_tree(self, data_split_name, p, m):
        """
        Function to construct a frequency tree

        The frequency tree is the column occurrence of 0 vs 1 for each column, normalized by the class occurrence. The
            function also uses a smoothing function, defined as: (Column Occurrence + (m * p)) / (Class Occurence + m)

        :param data_split_name: str, train or tune. Specifies the data split name to use
        :param p: float, value for smoothing function
        :param m: float, value for smoothing function
        """
        # Since only a fit call will trigger this function, we'll reinitialize the tree
        self.variable_frequency = {}

        # Grab data attributes
        data = self.data_split[data_split_name]
        overall_normalizer = len(data)

        # For each class, we'll construct a separate tree
        for class_name in self.class_names:
            # The normalizer is the count occurrence of the class
            class_normalizer = len(data.loc[data['Class'] == class_name])

            # This is the base dictionary of the class, with the normalizer and the %
            frequency_dict = {
                'class_normalizer': class_normalizer,
                'class_frequency': class_normalizer / overall_normalizer
            }

            # Now we'll go through each column and create two children per column, 0 and 1
            for column_name in data.keys()[:-1]:

                # Update the frequency dictionary with column_name, 0 child, and 1 child
                # The 0 and 1 children use the smoothing formula documented above
                frequency_dict.update({
                    column_name: {
                        0: (len(data.loc[(data[column_name] == 0) & (data['Class'] == class_name)]) +
                            (m * p)) /
                           (class_normalizer + m),
                        1: (len(data.loc[(data[column_name] == 1) & (data['Class'] == class_name)]) +
                            (m * p)) /
                           (class_normalizer + m)
                    }
                })

            # Once done with the class we'll update the base tree
            self.variable_frequency.update({class_name: frequency_dict})

    def classify(self, data_split_name):
        """
        Function to classify results using the frequency tree

        This function can be used by the fit or predict functions after a frequency tree has been built. It calculates
            a class coefficient per class for each row, and assigns the class based on the highest coefficient.
            The class coefficient comes from multiplying the value of the column * the class node in the frequency tree

        :param data_split_name: str, train or tune. Specifies the data split name to use
        :return results[0]: float, accuracy score for this classification
        :return results[1]: DataFrame, results of the classification appended onto the data used
        """
        # Initialize our data
        data = self.data_split[data_split_name]
        classification_list = []

        # Accuracy metrics
        true_positive = 0

        # We'll go through each row in the data
        for index, row in data.iterrows():
            # Each new row will initialize an class_coefficient dictionary starting with 1s for all class names
            class_coefficient = {key: 1 for (key) in self.class_names}

            # For each class we'll calculate a class_coefficient and update the above dictionary
            for class_name in self.class_names:
                # For each column in the row, we'll update the class_coefficient using the value * the frequency node
                # To travel down the frequency tree we use class_name -> column_name -> column value
                for column_name in data.keys()[:-1]:
                    class_coefficient[class_name] = class_coefficient[class_name] *\
                                                    self.variable_frequency[class_name][column_name][row[column_name]]

            # Retrieve the class name key with the highest coefficient and add to our list
            classification = max(class_coefficient, key=class_coefficient.get)
            classification_list.append(classification)

            # If our classification was correct we'll add a to our accuracy metric
            if classification == row['Class']:
                true_positive += 1

        # Accuracy calculation
        accuracy = true_positive / len(data)

        # Let's also go back to the original untransformed data set and attach our predictions there
        train_result_df = pd.DataFrame(self.data, index=data.index)  # TODO this is broken for breast-cancer
        train_result_df['prediction'] = classification_list

        return accuracy, train_result_df

    def fit(self, data_split_name='train', p=None, m=None):
        """
        Fit function

        This function pulls together the frequency tree construction and the classification. P and M can be passed here
            onto the frequency tree construction

        :param data_split_name: str, train or tune. Specifies the data split name to use
        :param p: float, value for smoothing function
        :param m: float, value for smoothing function
        :return accuracy: float, accuracy score for the classification
        """
        # If no parameters grab the object's current theta and alpha
        if p == None:
            p = self.p
        if m == None:
            m = self.m

        # First we need to construct the frequency tree
        self.construct_frequency_tree(data_split_name, p=p, m=m)

        # Use the tree to classify, getting back an accuracy score and the results of the classification
        results = self.classify(data_split_name)

        # If we are using the train data set we need to save the data back to our object
        if data_split_name == 'train':
            self.train_accuracy = results[0]
            self.train_results = results[1]

        return results[0]

    def tune(self):
        """
        Tune function

        This function uses the fit function targeted at the tune data split. It loops through a predefined list of p
            and m in order to determine the highest accuracy score.
        """
        # P and m ranges. P starts at .0002 and goes up by .0002 to .001. Alpha start at .25 and goes up by .25 to 1
        p_list = np.linspace(.0002, .001, 5)
        m_list = np.linspace(.25, 1, 4)

        # Variables for remembering the optimal p and m
        max_accuracy = 0
        optimal_p = .001
        optimal_m = 1

        # Go through each p and m (similar to a grid search)
        for p in p_list:
            for m in m_list:
                # Fit and retrieve the accuracy
                accuracy = self.fit(data_split_name='tune', p=p, m=m)

                # Check for most optimal p and m
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    optimal_p = p
                    optimal_m = m

                # Let's append the result back to the object for visualization later
                self.tune_parameter_list.append(f'{p}, {m}')
                self.tune_accuracy_list.append(accuracy)

        # After the loops are done, we'll set the model's p and m
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
        ax.set_title(rf'{self.data_name} Tune Results - Optimal: p {self.p}, m {self.m}')

        # X axis
        p_list = np.linspace(.0002, .001, 5)
        ax.set_xlabel(r'Parameters - Major Ticks: p, Minor Ticks: m')
        ax.set_xlim(4, 20)
        ax.set_xticks(np.linspace(0, 16, 5))
        ax.set_xticklabels(p_list, rotation=45, fontsize=6)
        ax.minorticks_on()

        # Y axis
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='y', which='minor', bottom=False)

        # Saving
        plt.savefig(f'output_{self.data_name}\\bayes_{self.data_name}_tune.jpg')

    def predict(self, data_split_name='test'):
        """
        Predict function

        This function uses only the classification function. So any previous frequency tree is used, as well as p and m

        :param data_split_name: str, train or tune. Specifies the data split name to use
        :return accuracy: float, accuracy score for the classification
        """
        # Classification
        results = self.classify(data_split_name)

        # Save the data back to our object
        self.test_accuracy = results[0]
        self.test_results = results[1]

        return results[0]

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
                'p': self.p,
                'm': self.m,
            },
            'train': {
                'accuracy': self.train_accuracy
            },
            'test': {
                'accuracy': self.test_accuracy
            }
        }

        # Saving
        with open(f'output_{self.data_name}\\bayes_{self.data_name}_summary.json', 'w') as file:
            json.dump(self.summary, file)

    def save_csv_results(self):
        """
        Function to output a csv of the results

        This uses the split of the original data set as output

        :return: csv to output folder
        """
        # Train
        self.train_results.to_csv(f'output_{self.data_name}\\bayes_{self.data_name}_train_results.csv')

        # Test
        self.test_results.to_csv(f'output_{self.data_name}\\bayes_{self.data_name}_test_results.csv')
