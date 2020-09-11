from utils.args import args
from etl.etl import ETL
from winnow2.multi_winnow2 import MultiWinnow2
from winnow2.winnow2 import Winnow2
from bayes.naive_bayes import NaiveBayes


def main():
    arguments = args()

    kwargs = {
        'data_name': arguments.data_name,
        'random_state': arguments.random_state
    }
    etl = ETL(**kwargs)
    if not arguments.naive_bayes:
        if etl.classes == 2:
            etl.train_test_split()

            winnow_model = Winnow2(etl)
            winnow_model.tune()
            winnow_model.visualize_tune()

            winnow_model.fit()

            winnow_model.predict()

            winnow_model.create_and_save_summary()
            winnow_model.save_csv_results()

        else:
            multi_winnow_model = MultiWinnow2(etl)

            multi_winnow_model.split_etl()

            multi_winnow_model.individual_class_winnow2()

            multi_winnow_model.multi_class_winnow2()

            multi_winnow_model.create_and_save_summary()
            multi_winnow_model.save_csv_results()

    else:
        etl.train_test_split()

        naive_bayes = NaiveBayes(etl)

        naive_bayes.tune()

        naive_bayes.visualize_tune()

        print(naive_bayes.fit())

    pass


if __name__ == '__main__':
    main()
