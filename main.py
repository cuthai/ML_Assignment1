from utils.args import args
from etl.etl import ETL
from winnow2.winnow2 import Winnow2


def main():
    arguments = args()

    kwargs = {
        'data_name': arguments.data_name
    }
    etl = ETL(**kwargs)

    Winnow2(etl)


if __name__ == '__main__':
    main()
