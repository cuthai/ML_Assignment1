import copy
from winnow2.winnow2 import Winnow2
import pandas as pd
import numpy as np


class MultiWinnow2:
    def __init__(self, etl):
        self.base_etl = etl
        self.base_etl.train_test_split()
        self.classes = self.base_etl.classes

        self.etl_list = {}
        self.winnow2_model_list = {}

        self.train_classification_coefficient_df = pd.DataFrame()
        self.test_classification_coefficient_df = pd.DataFrame()

        self.test_results = None
        self.train_results = None

        self.split_etl()
        self.individual_class_winnow2()
        self.multi_class_winnow2()

    def split_etl(self):
        class_list = list(range(1, self.classes + 1))
        class_list.reverse()

        for index in class_list:
            temp_etl = copy.deepcopy(self.base_etl)
            temp_x = temp_etl.transformed_data.iloc[:, :-self.classes]
            temp_y_name = temp_etl.transformed_data.keys()[-index]
            temp_y = temp_etl.transformed_data.iloc[:, -index]

            temp_x[temp_y_name] = temp_y

            temp_etl.transformed_data = temp_x

            temp_etl.train_test_split()
            temp_etl.update_data_name(temp_y_name)

            self.etl_list.update({temp_y_name: temp_etl})

    def individual_class_winnow2(self):
        for class_name in self.etl_list.keys():
            etl = self.etl_list[class_name]

            print(class_name)

            temp_winnow2_model = Winnow2(etl)
            temp_winnow2_model.tune()
            temp_winnow2_model.visualize_tune()

            train_results = temp_winnow2_model.fit()
            self.winnow2_model_list.update({class_name: temp_winnow2_model})
            self.train_classification_coefficient_df[class_name] = train_results[0]
            print(train_results[2])

            test_results = temp_winnow2_model.predict()
            self.test_classification_coefficient_df[class_name] = test_results[0]
            print(test_results[2])

    def multi_class_winnow2(self):
        train_order_df = np.argsort(-self.train_classification_coefficient_df.values, axis=1)
        train_result_df = pd.DataFrame(self.train_classification_coefficient_df.columns[train_order_df],
                                       index=self.train_classification_coefficient_df.index)[0]
        train_result_df = train_result_df.str.replace('Class_', '').to_list()
        self.train_results = pd.DataFrame.copy(self.base_etl.data.iloc[self.base_etl.data_split['train'].index],
                                               deep=True)
        self.train_results['Prediction'] = train_result_df

        test_order_df = np.argsort(-self.test_classification_coefficient_df.values, axis=1)
        test_result_df = pd.DataFrame(self.test_classification_coefficient_df.columns[test_order_df],
                                      index=self.test_classification_coefficient_df.index)[0]
        test_result_df = test_result_df.str.replace('Class_', '').to_list()
        self.test_results = pd.DataFrame.copy(self.base_etl.data.iloc[self.base_etl.data_split['test'].index],
                                              deep=True)
        self.test_results['Prediction'] = test_result_df
