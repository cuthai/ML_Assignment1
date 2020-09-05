from utils.args import args
from etl.etl import ETL
from winnow2.winnow2 import Winnow2


def main():
    arguments = args()

    kwargs = {
        'data_name': arguments.data_name
    }
    etl = ETL(**kwargs)

    winnow_model = Winnow2(etl)

    winnow_model.tune()

    print(winnow_model.fit())

    print(winnow_model.predict())


if __name__ == '__main__':
    main()
