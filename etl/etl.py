import pandas as pd


class ETL:
    """
    Class ETL to handle the ETL of the data.

    This class really only does the extract and transform functions of ETL. The data is then received downstream by the
    classifier algorithms for processing.
    """
    def __init__(self, data_name):
        """
        Init function. Takes a data_name and extracts the data and then transforms.

        All data comes from the data folder. The init function calls to both extract and transform for processing

        :param data_name: Str, name of the data file passed at the command line. Below are the valid names:
            breast-cancer
            glass
            iris
            soybean
            vote
        """
        # Set the starting attributes and data_name
        self.data = None
        self.data_name = data_name

        # Extract
        self.extract()

    def extract(self):
        if self.data_name == 'breast-cancer':
            column_names = ['ID', 'Clump_Thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape',
                            'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
                            'Normal_Nucleoli', 'Mitoses', 'Class']
            self.data = pd.read_csv('data\\breast-cancer-wisconsin.data', names=column_names)

        if self.data_name == 'glass':
            column_names = ['ID', 'Refractive_Index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium',
                            'Calcium', 'Barium', 'Iron', 'Class']
            self.data = pd.read_csv('data\\glass.data', names=column_names)
