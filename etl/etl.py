import pandas as pd
import numpy as np


class ETL:
    """
    Class ETL to handle the ETL of the data.

    This class really only does the extract and transform functions of ETL. The data is then received downstream by the
    classifier algorithms for processing.
    """
    def __init__(self, data_name, random_state=1):
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
        self.transformed_data = None
        self.data_split = {}
        self.data_name = data_name
        self.random_state = random_state
        self.classes = 0

        # Extract
        self.extract()

        # Transform
        self.transform()

    def extract(self):
        if self.data_name == 'breast-cancer':
            column_names = ['ID', 'Clump_Thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape',
                            'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
                            'Normal_Nucleoli', 'Mitoses', 'Class']
            self.data = pd.read_csv('data\\breast-cancer-wisconsin.data', names=column_names)

        elif self.data_name == 'glass':
            column_names = ['ID', 'Refractive_Index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium',
                            'Calcium', 'Barium', 'Iron', 'Class']
            self.data = pd.read_csv('data\\glass.data', names=column_names)

        elif self.data_name == 'iris':
            column_names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Class']
            self.data = pd.read_csv('data\\iris.data', names=column_names)

        elif self.data_name == 'soybean':
            column_names = ['Date', 'Plant_Stand', 'Percip', 'Temp', 'Hail', 'Crop_Hist', 'Area_Damaged', 'Severity',
                            'Seed_Tmt', 'Germination', 'Plant_Growth', 'Leaves', 'Leaf_Spots_Halo', 'Leaf_Spots_Marg',
                            'Leaf_Spot_Size', 'Leaf_Shread', 'Leaf_Malf', 'Leaf_Mild', 'Stem', 'Lodging',
                            'Stem_Cankers', 'Canker_Lesion', 'Fruiting_Bodies', 'External_Decay', 'Mycelium',
                            'Int_Discolor', 'Sclerotia', 'Fruit_Pods', 'Fruit_Spots', 'Seed', 'Mold_Growth',
                            'Seed_Discolor', 'Seed_Size', 'Shriveling', 'Roots', 'Class']
            self.data = pd.read_csv('data\\soybean-small.data', names=column_names)

        elif self.data_name == 'vote':
            column_names = ['Class', 'Handicapped_Infants', 'Water_Project_Cost_Sharing', 'Adoption_Budget_Resolution',
                            'Physician_Fee_Freeze', 'El_Salvador_Aid', 'Religious_Groups_School',
                            'Anti_Satellite_Test_Ban', 'Aid_Nicaraguan_Contras', 'MX_Missile', 'Immigration',
                            'Synfuels_Corporation_Cutback', 'Education_Spending', 'Superfund_Right_To_Sue', 'Crime',
                            'Duty_Free_Exports', 'Export_Administration_Act_South_Africa']
            self.data = pd.read_csv('data\\house-votes-84.data', names=column_names)

        else:
            raise NameError('Please specify a predefined name for one of the 5 data sets')

    def transform(self):
        if self.data_name == 'breast-cancer':
            self.transform_breast_cancer()

        elif self.data_name == 'glass':
            self.transform_glass()

        elif self.data_name == 'iris':
            self.transform_iris()

        elif self.data_name == 'soybean':
            self.transform_soybean()

        elif self.data_name == 'vote':
            self.transform_vote()

        else:
            raise NameError('Please specify a predefined name for one of the 5 data sets')

    def transform_breast_cancer(self):
        temp_df = pd.DataFrame.copy(self.data)
        temp_df = temp_df.loc[temp_df['Bare_Nuclei'] != '?']
        temp_df.drop(columns='ID', inplace=True)

        temp_df = pd.get_dummies(temp_df, columns=['Clump_Thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape',
                                                   'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei',
                                                   'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class'])
        temp_df.reset_index(inplace=True, drop=True)

        temp_df.drop(columns='Class_2', inplace=True)

        self.classes = 2
        self.transformed_data = temp_df

    def transform_glass(self):
        temp_df = pd.DataFrame.copy(self.data)
        temp_df.drop(columns='ID', inplace=True)

        for column in ['Refractive_Index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium',
                       'Barium', 'Iron']:
            temp_df[column] = pd.cut(temp_df[column], bins=10)

        temp_df = pd.get_dummies(temp_df, columns=['Refractive_Index', 'Sodium', 'Magnesium', 'Aluminum',
                                                   'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron', 'Class'])
        temp_df.reset_index(inplace=True, drop=True)

        self.classes = 6
        self.transformed_data = temp_df

    def transform_iris(self):
        temp_df = pd.DataFrame.copy(self.data)

        for column in ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']:
            temp_df[column] = pd.cut(temp_df[column], bins=10)

        temp_df = pd.get_dummies(temp_df, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width',
                                                   'Class'])
        temp_df.reset_index(inplace=True, drop=True)

        self.classes = 3
        self.transformed_data = temp_df

    def transform_soybean(self):
        temp_df = pd.DataFrame.copy(self.data)

        temp_df = pd.get_dummies(temp_df, columns=['Date', 'Plant_Stand', 'Percip', 'Temp', 'Hail', 'Crop_Hist',
                                                   'Area_Damaged', 'Severity', 'Seed_Tmt', 'Germination',
                                                   'Plant_Growth', 'Leaves', 'Leaf_Spots_Halo', 'Leaf_Spots_Marg',
                                                   'Leaf_Spot_Size', 'Leaf_Shread', 'Leaf_Malf', 'Leaf_Mild', 'Stem',
                                                   'Lodging', 'Stem_Cankers', 'Canker_Lesion', 'Fruiting_Bodies',
                                                   'External_Decay', 'Mycelium', 'Int_Discolor', 'Sclerotia',
                                                   'Fruit_Pods', 'Fruit_Spots', 'Seed', 'Mold_Growth', 'Seed_Discolor',
                                                   'Seed_Size', 'Shriveling', 'Roots', 'Class'])
        temp_df.reset_index(inplace=True, drop=True)

        self.classes = 4
        self.transformed_data = temp_df

    def transform_vote(self):
        temp_df = pd.DataFrame.copy(self.data)

        temp_df = pd.get_dummies(temp_df, columns=['Handicapped_Infants', 'Water_Project_Cost_Sharing',
                                                   'Adoption_Budget_Resolution', 'Physician_Fee_Freeze',
                                                   'El_Salvador_Aid', 'Religious_Groups_School',
                                                   'Anti_Satellite_Test_Ban', 'Aid_Nicaraguan_Contras', 'MX_Missile',
                                                   'Immigration', 'Synfuels_Corporation_Cutback', 'Education_Spending',
                                                   'Superfund_Right_To_Sue', 'Crime', 'Duty_Free_Exports',
                                                   'Export_Administration_Act_South_Africa', 'Class'])
        temp_df.reset_index(inplace=True, drop=True)

        temp_df.drop(columns='Class_democrat', inplace=True)

        self.classes = 2
        self.transformed_data = temp_df

    def train_test_split(self):
        data_size = len(self.transformed_data)
        tune_size = int(data_size / 10)
        train_size = int(data_size * (6 / 10))

        if self.random_state:
            np.random.seed(self.random_state)

        train_splitter = np.random.choice(a=data_size, size=train_size, replace=False)
        remainder = list(set(self.transformed_data.index) - set(train_splitter))
        tune_splitter = np.random.choice(a=remainder, size=tune_size, replace=False)
        test_splitter = list(set(remainder) - set(tune_splitter))

        if self.data_name == 'soybean':
            train_splitter = np.concatenate([train_splitter, tune_splitter])
            tune_splitter = train_splitter

        self.data_split.update({
            'train': self.transformed_data.iloc[train_splitter],
            'tune': self.transformed_data.iloc[tune_splitter],
            'test': self.transformed_data.iloc[test_splitter]
        })

    def update_data_name(self, class_name):
        self.data_name = f'{self.data_name}_{class_name}'
