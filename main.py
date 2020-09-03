from etl.etl import ETL
from utils.args import args


def main():
    arguments = args()

    kwargs = {
        'data_name': arguments.data_name
    }
    etl = ETL(**kwargs)

    pass


if __name__ == '__main__':
    main()
