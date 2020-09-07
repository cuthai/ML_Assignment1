from utils.args import args
from etl.etl import ETL
from winnow2.multi_winnow2 import MultiWinnow2
from winnow2.winnow2 import Winnow2


def main():
    arguments = args()

    kwargs = {
        'data_name': arguments.data_name
    }
    etl = ETL(**kwargs)
    if etl.classes == 2:
        etl.train_test_split()

        winnow_model = Winnow2(etl)
        winnow_model.tune()
        winnow_model.visualize_tune()

        train_results = winnow_model.fit()
        print(train_results[2])

        test_results = winnow_model.predict()
        print(test_results[2])

        winnow_model.save_results()

    else:
        multi_winnow_model = MultiWinnow2(etl)

        multi_winnow_model.split_etl()

        multi_winnow_model.individual_class_winnow2()

        multi_winnow_model.multi_class_winnow2()

        multi_winnow_model.save_results()


if __name__ == '__main__':
    main()
