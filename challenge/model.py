import pandas as pd

import numpy as np

from datetime import datetime

from typing import Tuple, Union, List

from sklearn.ensemble import RandomForestClassifier



class DelayModel:

    def __init__(
        self
    ):
        self._model = RandomForestClassifier() # Model should be saved in this attribute.

    def get_period_day(self, date):
        """
        Get period of the day based on the time of the day.

        Args:
            date (str): time of the day.

        Returns:
            (str): period of the day.
        """
        date_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
        morning_min = datetime.strptime("05:00", "%H:%M").time()
        morning_max = datetime.strptime("11:59", "%H:%M").time()
        afternoon_min = datetime.strptime("12:00", "%H:%M").time()
        afternoon_max = datetime.strptime("18:59", "%H:%M").time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()

        if (date_time >= morning_min and date_time <= morning_max):
            return 'mañana'
        elif (date_time > afternoon_min and date_time < afternoon_max):
            return 'tarde'
        elif (
                (date_time > evening_min and date_time < evening_max) or
                (date_time > night_min and date_time < night_max)
        ):
            return 'noche'

    def is_high_season(self, fecha):
        """
                Get if is a high season based on the date.

                Args:
                    date (str): time of the day.

                Returns:
                    (int): 1 if is High season, 0 if not.
                """
        fecha_anio = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_anio)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_anio)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_anio)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_anio)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_anio)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_anio)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_anio)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_anio)

        if ((fecha >= range1_min and fecha <= range1_max) or
                (fecha >= range2_min and fecha <= range2_max) or
                (fecha >= range3_min and fecha <= range3_max) or
                (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0

    def get_min_diff(self, data):
        """
                        Get the minutes difference based on time of the date of departure and arrival.

                        Args:
                            data (pd.DataFrame): raw data.

                        Returns:
                            (float): delay of the flight.
                        """
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60

        return min_diff

    def preprocess(self,
            data: pd.DataFrame,
        target_column: str = "None"
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data['period_day'] = data['Fecha-I'].apply(self.get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self.is_high_season)
        data['min_diff'] = data.apply(self.get_min_diff, axis=1)

        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        features = data.drop(target_column, axis=1)
        target = data[target_column]

        return features, target

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._model.fit(features, target)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        return self._model.predict(features).tolist()