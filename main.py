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
        print(winnow_model.fit())
        print(winnow_model.predict())

    else:
        winnow_model = MultiWinnow2(etl)

    pass


if __name__ == '__main__':
    main()