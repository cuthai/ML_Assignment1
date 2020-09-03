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

        if self.data_name == 'iris':
            column_names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Class']
            self.data = pd.read_csv('data\\iris.data', names=column_names)

        if self.data_name == 'soybean':
            column_names = ['Date', 'Plant_Stand', 'Percip', 'Temp', 'Hail', 'Crop_Hist', 'Area_Damaged', 'Severity',
                            'Seed_Tmt', 'Germination', 'Plant_Growth', 'Leaves', 'Leaf_Spots_Halo', 'Leaf_Spots_Marg',
                            'Leaf_Spot_Size', 'Leaf_Shread', 'Leaf_Malf', 'Leaf_Mild', 'Stem', 'Lodging',
                            'Stem_Cankers', 'Canker_Lesion', 'Fruiting_Bodies', 'External_Decay', 'Mycelium',
                            'Int_Discolor', 'Sclerotia', 'Fruit_Pods', 'Fruit_Spots', 'Seed', 'Mold_Growth',
                            'Seed_Discolor', 'Seed_Size', 'Shriveling', 'Roots', 'Class']
            self.data = pd.read_csv('data\\soybean-small.data', names=column_names)

        if self.data_name == 'vote':
            column_names = ['Class', 'Handicapped_Infants', 'Water_Project_Cost_Sharing', 'Adoption_Budget_Resolution',
                            'Physician_Fee_Freeze', 'El_Salvador_Aid', 'Religious_Groups_School',
                            'Anti_Satellite_Test_Ban', 'Aid_Nicaraguan_Contras', 'MX_Missile', 'Immigration',
                            'Synfuels_Corporation_Cutback', 'Education_Spending', 'Superfund_Right_To_Sue', 'Crime',
                            'Duty_Free_Exports', 'Export_Administration_Act_South_Africa']
            self.data = pd.read_csv('data\\house-votes-84.data', names=column_names)
