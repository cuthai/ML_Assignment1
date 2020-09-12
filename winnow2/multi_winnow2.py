import copy
from winnow2.winnow2 import Winnow2
import pandas as pd
import numpy as np
import json


class MultiWinnow2:
    """
    MultiWinnow2 Object to handle multi class data sets

    For the most part this object uses individual Winnow2 objects trained to each target class in the data set. The
        output classification_coefficients are then compared to take the highest score to assign for that class.
        This object has a fit and predict, but the actual tune is handled by the individual Winnow2 object
    """
    def __init__(self, etl):
        """
        Init function. Takes an etl object

        :param etl: etl, etl object with information and data from the data set
        """
        # ETL attributes
        self.base_etl = etl
        self.data_name = self.base_etl.data_name
        self.base_etl.train_test_split()
        self.classes = self.base_etl.classes

        # Attributes for holding multiple ETL and Winnow2 objects. We will be multiplying the data set for each class
        self.etl_list = {}
        self.winnow2_model_list = {}

        # Combined classification coefficients for train and test
        self.train_classification_coefficient_df = pd.DataFrame()
        self.test_classification_coefficient_df = pd.DataFrame()

        # Results
        self.test_results = None
        self.train_results = None

        # Summary
        self.summary = {}

    def split_etl(self):
        """
        This function splits the etl object by multiplying it

        For each etl, it picks one class, and sets that etl to target that class. In this way, the original etl is
            duplicated for the number of classes. Each class will then have an individual Winnow2 object. This function
            primarily modifies the etl for the individual Winnow2
        """
        # Let's determine the number of classes
        class_list = list(range(1, self.classes + 1))
        class_list.reverse()

        # For each index in the class list, we will be pulling out the y and creating a new ETL
        for index in class_list:
            # Let's make a deep copy of our original ETL
            temp_etl = copy.deepcopy(self.base_etl)

            # Resolving x and y temporarily. y is the single class by the index for this loop
            temp_x = temp_etl.transformed_data.iloc[:, :-self.classes]
            temp_y_name = temp_etl.transformed_data.keys()[-index]
            temp_y = temp_etl.transformed_data.iloc[:, -index]

            # Join single y back onto x
            temp_x[temp_y_name] = temp_y

            # Attaching the new data set onto the transformed data attribute of the etl object
            temp_etl.transformed_data = temp_x

            # Now that we have a single target class, we need to split. Random state is handled with the base etl so all
            # splits for each new etl object will be the same
            temp_etl.train_test_split()

            # Let's update the data name to include the class name
            temp_etl.update_data_name(temp_y_name)

            self.etl_list.update({temp_y_name: temp_etl})

    def individual_class_winnow2(self):
        """
        Function to synthesize the results of multiple Winnow2 objects trained on an individual class

        This function uses the ETLs created in the split_etl function. It creates and trains a Winnow2 object on each
            ETL which was split to have one a single target class
        """
        # Loop through each individual ETL
        for class_name in self.etl_list.keys():
            # Define our ETL
            etl = self.etl_list[class_name]

            # Create a Winnow2 model and then tune as well as visualize it
            temp_winnow2_model = Winnow2(etl)
            temp_winnow2_model.tune()
            temp_winnow2_model.visualize_tune()

            # Fit
            train_results = temp_winnow2_model.fit()

            # Add the fit object back to the Winnow2_Model_List
            self.winnow2_model_list.update({class_name: temp_winnow2_model})

            # Retrieve the train classification coefficients to add to the classification_coefficient_df
            self.train_classification_coefficient_df[class_name] = train_results[0]

            # Predict
            test_results = temp_winnow2_model.predict()

            # Retrieve the test classification coefficients to add to the classification_coefficient_df
            self.test_classification_coefficient_df[class_name] = test_results[0]

            # Let's also save the csv results and summary of the individual class
            temp_winnow2_model.create_and_save_summary()
            temp_winnow2_model.save_csv_results()

    def multi_class_winnow2(self):
        """
        Function to determine highest scoring class

        The function uses the classification coefficient df for all the Winnow2 objects to pick a single class
        """
        # Train
        # We'll reorder the index of the classification_coefficient_df to grab the highest scoring index
        train_order_df = np.argsort(-self.train_classification_coefficient_df.values, axis=1)

        # That index is then used to grab the column names, which are the class names
        train_result_df = pd.DataFrame(self.train_classification_coefficient_df.columns[train_order_df],
                                       index=self.train_classification_coefficient_df.index)[0]

        # Remove Class_ from the object
        train_result_df = train_result_df.str.replace('Class_', '').to_list()

        # Grab the original DataFrame of our base ETL
        self.train_results = pd.DataFrame.copy(self.base_etl.data.iloc[self.base_etl.data_split['train'].index],
                                               deep=True)

        # Add the predictions onto that
        self.train_results['Prediction'] = train_result_df

        # Test
        # We'll reorder the index of the classification_coefficient_df to grab the highest scoring index
        test_order_df = np.argsort(-self.test_classification_coefficient_df.values, axis=1)

        # That index is then used to grab the column names, which are the class names
        test_result_df = pd.DataFrame(self.test_classification_coefficient_df.columns[test_order_df],
                                      index=self.test_classification_coefficient_df.index)[0]

        # Remove Class_ from the object
        test_result_df = test_result_df.str.replace('Class_', '').to_list()

        # Grab the original DataFrame of our base ETL
        self.test_results = pd.DataFrame.copy(self.base_etl.data.iloc[self.base_etl.data_split['test'].index],
                                              deep=True)

        # Add the predictions onto that
        self.test_results['Prediction'] = test_result_df

        # For glass, the class names are just integers so we need to convert that from an object to an int
        if self.data_name == 'glass':
            self.train_results['Prediction'] = pd.to_numeric(self.train_results['Prediction'])
            self.test_results['Prediction'] = pd.to_numeric(self.test_results['Prediction'])

    def create_and_save_summary(self):
        """
        Function to create a summary

        Creates a JSON summary for this object and outputs a JSON document to the output folder. This function first
            creates an overall summary for all the individual Winnow2 objects, before adding summaries for each Winnow2
            and their target class

        :return: JSON to output folder
        """
        # Overall Accuracy calculation
        train_accuracy = len(
            self.train_results.loc[
                self.train_results['Class'] == self.train_results['Prediction']]) / len(self.train_results)
        test_accuracy = len(
            self.test_results.loc[
                self.test_results['Class'] == self.test_results['Prediction']]) / len(self.test_results)

        # Overall
        self.summary.update({
            'Overall': {
                'train': train_accuracy,
                'test': test_accuracy
            }
        })

        # Summary of individual Winnow2
        for winnow2_model in self.winnow2_model_list.values():
            summary = {
                winnow2_model.data_name: {
                    'tune': {
                        'theta': winnow2_model.theta,
                        'alpha': winnow2_model.alpha
                    },
                    'train': {
                        'accuracy': winnow2_model.train_accuracy
                    },
                    'test': {
                        'accuracy': winnow2_model.test_accuracy
                    }
                }
            }

            self.summary.update(summary)

        # Saving
        with open(f'output_{self.data_name}\\winnow_{self.data_name}_summary.json', 'w') as file:
            json.dump(self.summary, file)

    def save_csv_results(self):
        """
        Function to output a csv of the results

        This uses the split of the original data set as output

        :return: csv to output folder
        """
        # Train
        self.train_results.to_csv(f'output_{self.data_name}\\winnow_{self.data_name}_train_results.csv')

        # Test
        self.test_results.to_csv(f'output_{self.data_name}\\winnow_{self.data_name}_test_results.csv')
