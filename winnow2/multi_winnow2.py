import copy
from winnow2.winnow2 import Winnow2


class MultiWinnow2:
    def __init__(self, etl):
        self.base_etl = etl
        self.classes = self.base_etl.classes

        self.etl_list = {}
        self.winnow2_model_list = {}

        self.split_etl()

        self.individual_class_winnow2()

        pass

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

            self.etl_list.update({temp_y_name: temp_etl})

    def individual_class_winnow2(self):
        for class_name in self.etl_list.keys():
            etl = self.etl_list[class_name]

            print(class_name)

            temp_winnow2_model = Winnow2(etl)
            temp_winnow2_model.tune()
            print(temp_winnow2_model.fit())
            print(temp_winnow2_model.predict())

            self.winnow2_model_list.update({class_name: temp_winnow2_model})
