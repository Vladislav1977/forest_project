# OVERVIEW

The project is based on Forest Cover Type dataset, currently running on [Kaggle competition](https://www.kaggle.com/competitions/forest-cover-type-prediction/overview).

The aim of this project:
- test some ML techniques (Logistic Regression, SVM, Gradient Boosting,  Extremely Randomized Trees and FC NN) to achieve high prediction score.
- implement convenient and simple way for training above techniques and making their predictions.

# FEATURE ENGINEERING

The main part playing incredible role in model evaluation.

`Soil_Type15`, `Soil_Type7` columns were dropped.
Two set of features were made. The first one with the most important features. And the second one with additional features added to the previous set.

<table>
  <tr>
    <td>elev_hydro_vert_sum</td>
    <td>Elevation + Vertical_Distance_To_Hydrology</td>
  </tr>
  <tr>
    <td>elev_hydro_vert_diff</td>
    <td>abs(Elevation - Vertical_Distance_To_Hydrology)</td>
  </tr>
    <tr>
    <td>dist_to_hydrology</td>
    <td>sqrt(Horizontal_Distance_To_Hydrology ** 2 + Vertical_Distance_To_Hydrology ** 2)</td>
   </tr>
    <tr>
    <td>dist_to_hydrology_squared</td>
    <td>dist_to_hydrology ** 2</td>
  </tr>
      <tr>
    <td>cos_to_hydr</td>
    <td> Horizontal_Distance_To_Hydrology / (dist_to_hydrology + 1e-15)</td>
  </tr>
    <tr>
    <td>mean_amenities</td>
    <td> (Horizontal_Distance_To_Hydrology + Horizontal_Distance_To_Roadways + Horizontal_Distance_To_Fire_Points) / 3  </td>
  </tr>
  <tr>
    <td>mean_amenities_squared_root</td>
    <td> sqrt(Horizontal_Distance_To_Hydrology ** 2 + Horizontal_Distance_To_Roadways ** 2 + Horizontal_Distance_To_Fire_Points ** 2) </td>
  </tr>
    <tr>
    <td>fire_road_sum_squared</td>
    <td> (Horizontal_Distance_To_Fire_Points + Horizontal_Distance_To_Roadways) ** 2 </td>
  </tr>
  <tr>
    <td>fire_road_diff_squared</td>
    <td> (Horizontal_Distance_To_Fire_Points - Horizontal_Distance_To_Roadways) ** 2</td>
  </tr>
  <tr>
    <td>hydro_road_sum_squared</td>
    <td> (Horizontal_Distance_To_Hydrology + Horizontal_Distance_To_Roadways) ** 2</td>
  </tr>
    <tr>
    <td>hydro_road_diff_squared</td>
    <td>  (Horizontal_Distance_To_Hydrology - Horizontal_Distance_To_Roadways) ** 2 </td>
  </tr>
  <tr>
    <td>fire_hydro_sum_squared</td>
    <td> (Horizontal_Distance_To_Fire_Points + Horizontal_Distance_To_Hydrology) ** 2</td>
  </tr>
    <tr>
    <td>fire_hydro_diff_squared</td>
    <td> (Horizontal_Distance_To_Fire_Points - Horizontal_Distance_To_Hydrology) ** 2</td>
  </tr>
  <tr>
    <td>vert_below</td>
    <td> Vertical_Distance_To_Hydrology.map(lambda x: 0 if x < 0 else 1) </td>
  </tr>
   <tr>
    <td colspan="2">Second set </td>
  </tr>
    <tr>
    <td>fire_road_sum</td>
    <td> (Horizontal_Distance_To_Fire_Points + Horizontal_Distance_To_Roadways) </td>
  </tr>
  <tr>
    <td>fire_road_diff</td>
    <td> abs(Horizontal_Distance_To_Fire_Points - Horizontal_Distance_To_Roadways) </td>
  </tr>
  <tr>
    <td>hydro_road_sum</td>
    <td>Horizontal_Distance_To_Hydrology + Horizontal_Distance_To_Roadways</td>
  </tr>
  <tr>
    <td>hydro_road_diff</td>
    <td>abs(Horizontal_Distance_To_Hydrology - Horizontal_Distance_To_Roadways)</td>
  </tr>
    <tr>
    <td>fire_hydro_sum</td>
    <td> (Horizontal_Distance_To_Fire_Points + Horizontal_Distance_To_Hydrology)</td>
  </tr>
  <tr>
    <td>fire_hydro_diff</td>
    <td> abs(Horizontal_Distance_To_Fire_Points - x.Horizontal_Distance_To_Hydrology) </td>
  </tr>
<tr>
    <td>aspect3</td>
    <td> sin(Aspect) </td>
  </tr>
</table>

Numeric features were standardized by removing the mean and scaling to unit variance.

# Results

The best scores for each model are shown below.

<table>
  <tr>
    <td> </td>
    <td>cross_val score</td>
    <td>test score</td>
  </tr>
  <tr>
    <td> Logistic Regression (X_3) </td>
    <td>0.6669312169312169  </td>
    <td>0.60251</td>
  </tr>
  <tr>
    <td> Logistic Regression (X_3) </td>
    <td>0.6669312169312169 </td>
    <td>0.60251</td>
  </tr>
    <tr>
      <td> SVM (X_2) </td>
      <td>0.778968253968254 </td>
      <td>0.76968</td>
  </tr>
      <tr>
      <td> gradboost (X_2) </td>
      <td>0.8843253968253968</td>
      <td>0.78084</td>
  </tr>
        <tr>
      <td> gradboost (X_2) </td>
      <td>0.8220238095238095</td>
      <td>0.81067</td>
  </tr>
<tr>
      <td> FCNN (X_3) </td>
      <td>0.8547485375404358</td>
      <td>0.73174</td>
  </tr>
</table>
