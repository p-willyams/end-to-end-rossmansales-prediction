import pickle
import pandas as pd
import numpy as np
import datetime

class Rossman:
    """
    Classe principal para gerenciamento do modelo Rossman, incluindo limpeza de dados,
    engenharia de atributos, preparação dos dados e predição.

    Esta classe encapsula todo o pipeline de preparação e inferência, utilizando os mesmos 
    scalers e encoders salvos durante o treinamento para garantir consistência no deploy.
    """

    # Lista das features esperadas pelo modelo treinado
    expected_features = [
        'store',
        'promo',
        'competition_distance',
        'promo2',
        'competition_time_month',
        'promo_time_week',
        'month_cos',
        'month_sin',
        'day_sin',
        'day_cos',
        'day_of_week_sin',
        'day_of_week_cos',
        'week_of_year_sin',
        'week_of_year_cos',
        'store_type_encoded',
        'assortment_encoded'
    ]

    def __init__(self):
        with open('parameter/robust_scaler_competition_distance.pkl', 'rb') as f:
            self.scaler_competition_distance = pickle.load(f)
        with open('parameter/robust_scaler_competition_time_month.pkl', 'rb') as f:
            self.scaler_competition_time_month = pickle.load(f)
        with open('parameter/minmax_scaler_promo_time_week.pkl', 'rb') as f:
            self.scaler_promo_time_week = pickle.load(f)
        with open('parameter/minmax_scaler_year.pkl', 'rb') as f:
            self.scaler_year = pickle.load(f)
        with open('parameter/ohe_state_holiday.pkl', 'rb') as f:
            self.ohe_state_holiday = pickle.load(f)
        with open('parameter/le_store_type.pkl', 'rb') as f:
            self.le_store_type = pickle.load(f)
        with open('parameter/assortment_mapping.pkl', 'rb') as f:
            self.assortment_mapping = pickle.load(f)
        # Carrega o modelo treinado
        with open('model/model_rossman_sales.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
    def clean_data(self, df):
        df = df.copy()
        def snake_case(lst):
            def convert(s):
                s = s.replace(' ', '_')
                new_s = ""
                for i, c in enumerate(s):
                    if c.isupper():
                        if i > 0 and (s[i-1].islower() or (i+1 < len(s) and s[i+1].islower())):
                            new_s += "_"
                        new_s += c.lower()
                    else:
                        new_s += c
                return new_s
            return [convert(s) for s in lst]
        df.columns = snake_case(df.columns)
        df['date'] = pd.to_datetime(df['date'])
        df['competition_distance'] = df['competition_distance'].fillna(200000)
        df['competition_open_since_month'] = df['competition_open_since_month'].fillna(df['date'].dt.month)
        df['competition_open_since_year'] = df['competition_open_since_year'].fillna(df['date'].dt.year)
        df['promo2_since_week'] = df['promo2_since_week'].fillna(df['date'].dt.isocalendar().week)
        df['promo2_since_year'] = df['promo2_since_year'].fillna(df['date'].dt.year)
        df['promo_interval'] = df['promo_interval'].fillna(0)
        df['competition_open_since_month'] = df['competition_open_since_month'].astype(int)
        df['competition_open_since_year'] = df['competition_open_since_year'].astype(int)
        df['promo2_since_week'] = df['promo2_since_week'].astype(int)
        df['promo2_since_year'] = df['promo2_since_year'].astype(int)
        return df

    def feature_engineering(self, df):
        df = df.copy()
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        df['month_map'] = df['date'].dt.month.map(month_map)
        df['is_promo'] = df.apply(lambda row: 1 if str(row['month_map']) in str(row['promo_interval']) else 0, axis=1)
        df.drop('month_map', axis=1, inplace=True)
        df['year'] = df['date'].dt.year.astype(int)
        df['month'] = df['date'].dt.month.astype(int)
        df['day'] = df['date'].dt.day.astype(int)
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['competition_since'] = pd.to_datetime(
            dict(year=df['competition_open_since_year'],
                 month=df['competition_open_since_month'],
                 day=1)
        )
        df['competition_time_month'] = ((df['date'] - df['competition_since']).dt.days / 30).round().astype(int)
        df['promo_since'] = df['promo2_since_year'].astype(str) + '-' + df['promo2_since_week'].astype(str)
        df['promo_since'] = df['promo_since'].apply(
            lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7)
        )
        df['promo_time_week'] = ((df['date'] - df['promo_since']) / 7).apply(lambda x: x.days).astype(int)
        if 'assortment' in df.columns:
            df['assortment'] = df['assortment'].map({'a': 'basic', 'b': 'extra', 'c': 'extended'})
        if 'state_holiday' in df.columns:
            df['state_holiday'] = df['state_holiday'].map(
                {'a': 'public holiday', 'b': 'easter holiday', 'c': 'christmas', '0': 'regular_day'}
            )
        if 'month_cos' not in df.columns:
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        if 'month_sin' not in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        if 'day_cos' not in df.columns:
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        if 'day_sin' not in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        if 'day_of_week' in df.columns:
            if 'day_of_week_cos' not in df.columns:
                df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            if 'day_of_week_sin' not in df.columns:
                df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        if 'week_of_year_cos' not in df.columns:
            df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        if 'week_of_year_sin' not in df.columns:
            df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        drop_cols = [
            'customers',
            'open',
            'promo_interval',
            'promo2_since_year',
            'promo2_since_week',
            'competition_open_since_month',
            'competition_open_since_year',
            'competition_since',
            'promo_since'
        ]
        drop_cols = [col for col in drop_cols if col in df.columns]
        if drop_cols:
            df = df.drop(drop_cols, axis=1)
        return df

    def data_preparation(self, df):
        df = df.copy()
        if 'competition_distance' in df.columns:
            df['competition_distance'] = self.scaler_competition_distance.transform(df[['competition_distance']])
        if 'competition_time_month' in df.columns:
            df['competition_time_month'] = self.scaler_competition_time_month.transform(df[['competition_time_month']])
        if 'promo_time_week' in df.columns:
            df['promo_time_week'] = self.scaler_promo_time_week.transform(df[['promo_time_week']])
        if 'year' in df.columns:
            df['year'] = self.scaler_year.transform(df[['year']])
        if 'state_holiday' in df.columns:
            state_holiday_encoded = self.ohe_state_holiday.transform(df[['state_holiday']])
            state_holiday_columns = self.ohe_state_holiday.get_feature_names_out(['state_holiday'])
            df_state_holiday = pd.DataFrame(state_holiday_encoded, columns=state_holiday_columns, index=df.index)
            df = pd.concat([df.drop('state_holiday', axis=1), df_state_holiday], axis=1)
        if 'store_type' in df.columns:
            df['store_type_encoded'] = self.le_store_type.transform(df['store_type'])
        if 'assortment' in df.columns:
            df['assortment_encoded'] = df['assortment'].map(self.assortment_mapping)
        drop_cats = [c for c in ['assortment', 'store_type'] if c in df.columns]
        if drop_cats:
            df = df.drop(drop_cats, axis=1)
        missing_cols = [col for col in self.expected_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected features: {missing_cols}")
        df = df[self.expected_features]
        return df

    def predict(self, df):
        """
        Executa o pipeline completo: limpeza, feature engineering, preparação dos dados e retorna as predições em um DataFrame.
        
        Parâmetros:
        ----------
        df : pd.DataFrame
            DataFrame bruto de entrada.

        Retorno:
        -------
        df_predictions : pd.DataFrame
            DataFrame com os valores previstos de vendas para as amostras fornecidas.
        """
        # Limpa os dados
        df_clean = self.clean_data(df)
        # Engenharia de features
        df_feat = self.feature_engineering(df_clean)
        # Prepara os dados para o modelo
        df_ready = self.data_preparation(df_feat)
        # Faz predição
        preds = self.model.predict(df_ready)
        preds_exp = np.expm1(preds)
        df['prediction'] = preds_exp
        return df

