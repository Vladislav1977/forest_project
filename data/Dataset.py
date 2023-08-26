import numpy as np
import pandas as pd
import os, sys

class MyDataset:

    def __init__(self, path_train, path_test):
        df_train = pd.read_csv(path_train)
        df_test = pd.read_csv(path_test)


        self.Id = df_test["Id"]

        self.X_1 = df_train.drop(columns=["Id", "Soil_Type15", "Soil_Type7"]).iloc[:, :-1]
        self.X_1_test = df_test.drop(columns=["Id", "Soil_Type15", "Soil_Type7"])

        self.y = df_train.drop(columns=["Id", "Soil_Type15", "Soil_Type7"]).iloc[:, -1].to_numpy()

        self.X_2, self.X_2_test = self.feat_eng(mode=1)


        self.X_3, self.X_3_test = self.feat_eng(mode=2)

    def feat_eng(self, mode):

        df_train_eng = self.X_1.iloc[:, 0:10].copy()
        df_test_eng = self.X_1_test.iloc[:, 0:10].copy()

        frames = [df_train_eng, df_test_eng]
        for x in frames:
            x["elev_hydro_vert_sum"] = x.Elevation + x.Vertical_Distance_To_Hydrology
            x["elev_hydro_vert_diff"] = np.abs(x.Elevation - x.Vertical_Distance_To_Hydrology)
            x["elev_hydro_horr_sum"] = x.Elevation + x.Horizontal_Distance_To_Hydrology
            x["elev_hydro_horr_diff"] = np.abs(x.Elevation - x.Horizontal_Distance_To_Hydrology)
            x["dist_to_hydrology"] = np.sqrt(
                x.Horizontal_Distance_To_Hydrology ** 2 + x.Vertical_Distance_To_Hydrology ** 2)
            x["dist_to_hydrology_squared"] = x["dist_to_hydrology"] ** 2
            x["cos_to_hydr"] = x.Horizontal_Distance_To_Hydrology / (x.dist_to_hydrology + 1e-15)
            x["mean_amenities"] = (x.Horizontal_Distance_To_Hydrology + x.Horizontal_Distance_To_Roadways + x.Horizontal_Distance_To_Fire_Points) / 3
            x["mean_amenities_squared_root"] = np.sqrt(x.Horizontal_Distance_To_Hydrology ** 2 +
                                                       x.Horizontal_Distance_To_Roadways ** 2 +
                                                       x.Horizontal_Distance_To_Fire_Points ** 2)
            x["fire_road_sum_squared"] = (x.Horizontal_Distance_To_Fire_Points + x.Horizontal_Distance_To_Roadways) ** 2
            x["fire_road_diff_squared"] = (x.Horizontal_Distance_To_Fire_Points - x.Horizontal_Distance_To_Roadways) ** 2
            x["hydro_road_sum_squared"] = (x.Horizontal_Distance_To_Hydrology + x.Horizontal_Distance_To_Roadways) ** 2
            x["hydro_road_diff_squared"] = (x.Horizontal_Distance_To_Hydrology - x.Horizontal_Distance_To_Roadways) ** 2
            x["fire_hydro_sum_squared"] = (x.Horizontal_Distance_To_Fire_Points + x.Horizontal_Distance_To_Hydrology) ** 2
            x["fire_hydro_diff_squared"] = (x.Horizontal_Distance_To_Fire_Points - x.Horizontal_Distance_To_Hydrology) ** 2

            x["vert_below"] = x.Vertical_Distance_To_Hydrology.map(lambda x: 0 if x < 0 else 1)

            if mode == 2:
                x["fire_road_sum"] = (x.Horizontal_Distance_To_Fire_Points + x.Horizontal_Distance_To_Roadways)
                x["fire_road_diff"] = np.abs(
                    x.Horizontal_Distance_To_Fire_Points - x.Horizontal_Distance_To_Roadways)
                x["hydro_road_sum"] = (x.Horizontal_Distance_To_Hydrology + x.Horizontal_Distance_To_Roadways)
                x["hydro_road_diff"] = np.abs(
                    x.Horizontal_Distance_To_Hydrology - x.Horizontal_Distance_To_Roadways)
                x["fire_hydro_sum"] = (x.Horizontal_Distance_To_Fire_Points + x.Horizontal_Distance_To_Hydrology)
                x["fire_hydro_diff"] = np.abs(
                    x.Horizontal_Distance_To_Fire_Points - x.Horizontal_Distance_To_Hydrology)
                x["aspect3"] = np.sin(x.Aspect)

        df_train_eng = pd.concat([df_train_eng, self.X_1.iloc[:, 10:].copy()], axis=1)
        df_test_eng = pd.concat([df_test_eng, self.X_1_test.iloc[:, 10:].copy()], axis=1)

        return df_train_eng, df_test_eng

if __name__ == "__main__":
    path_train = r"../dataset/train.csv"
    path_test = r"../dataset/test.csv"

    df = MyDataset(path_train, path_test)
    print(df.X_2.shape)





