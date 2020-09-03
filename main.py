from etl.etl import ETL


def main():
    args = {
        'data_name': 'glass'
    }
    etl = ETL(**args)

    pass


if __name__ == '__main__':
    main()
