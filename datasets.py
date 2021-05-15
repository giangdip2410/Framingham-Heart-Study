import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class Datasets:
    """Dataset for classification problem"""

    def __init__(
        self,
        data_file="./train.csv",
        cat_cols=None,
        num_cols=None,
        level_cols=None,
        label_col=None,
        train=True,
    ):
        """create new copies instead of references"""
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.level_cols = level_cols
        self.feature_cols = cat_cols + num_cols + level_cols
        self.label_col = label_col
        self.label_encoder = None
        self.data_df = self._create_data_df(data_file)
        self.feature_train
        self.target_train
        self.feature_test
        self.target_test
        self.train = train
        self.X
        self.y
        self.scaler
        self.one_hot

    def _label_encode(self, df, col):
        """label encodes data"""
        le = LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
        self.label_encoder = le
        return df

    def _inverse_label_encode(self, df, col):
        """inverse label encodes data"""
        le = self.label_encoder
        df[col] = le.inverse_transform(df[col])

    def _load_data(self, file):
        """loads csv to pd dataframe"""
        return pd.read_csv(file)

    # def _create_kfold(self, file):
    #     """make k fold for data"""
    #     df = _load_data(file)
    #     df["kfold"] = -1

    #     df = df.sample(frac=1).reset_index(drop=True)

    #     kf = model_selection.StratifiedKFold(
    #         n_splits=self.kfold, shuffle=False, random_state=24
    #     )

    #     for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
    #         print(len(train_idx), len(val_idx))
    #         df.loc[val_idx, "kfold"] = fold
    #     return df

    def _create_data_df(self, data_file, preprocess=True, label_encode=False):
        """loads and encodes train data"""
        data = self._load_data(data_file)
        if preprocess:
            data = self._impute_missing_values(
                data, self.cat_cols, self.num_cols, self.level_cols
            )
            data = self._feature_preprocessing(
                data, self.cat_cols, self.num_cols, self.level_cols
            )
        if label_encode:
            self._label_encode(data, self.label_col)
        self._split_train_test(data)
        return data

    def _impute_missing_values(
        self, df, categorical_features, numeric_features, level_features
    ):
        """Imputes the continious columns with median and categorical columns with the mode value"""
        imputer_con = SimpleImputer(missing_values=np.nan, strategy="median")
        imputer_cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        for col in categorical_features + numeric_features + level_features:
            if df[col].isnull().sum() > 0:
                if col in categorical_features + level_features:
                    df[col] = imputer_cat.fit_transform(df[col].values.reshape(-1, 1))
                elif col in numeric_features:
                    df[col] = imputer_con.fit_transform(df[col].values.reshape(-1, 1))
        return df

    def _onehot_encoding(self, df, cat_features):
        encoded_features = []
        self.one_hot = {}
        for feature in cat_features:
            oh = OneHotEncoder()
            encoded_feat = oh.fit_transform(df[feature].values.reshape(-1, 1)).toarray()
            self.one_hot[feature] = oh
            n = df[feature].nunique()
            cols = ["{}_{}".format(feature, n) for n in range(1, n + 1)]
            self.one_hot[str(feature) + "col"] = cols
            encoded_df = pd.DataFrame(encoded_feat, columns=cols)
            encoded_df.index = df.index
            encoded_features.append(encoded_df)

        df = pd.concat([df, *encoded_features[:6]], axis=1)
        # drop columns after one hot
        df.drop(columns=cat_features, inplace=True)
        return df

    def _onehot_newdata(self, df):
        encoded_features = []
        for feature in self.cat_cols:
            oh = self.one_hot[feature]
            encoded_feat = oh.transform(df[feature].values.reshape(-1, 1)).toarray()
            self.one_hot[feature] = oh

            encoded_df = pd.DataFrame(
                encoded_feat, columns=self.one_hot[str(feature) + "col"]
            )
            encoded_df.index = df.index
            encoded_features.append(encoded_df)

        df = pd.concat([df, *encoded_features[:6]], axis=1)
        # drop columns after one hot
        df.drop(columns=self.cat_cols, inplace=True)
        # print(df)
        return df

    def _feature_preprocessing(self, df, cat_cols, num_cols, level_col):
        """This function preprocessing feature before training"""
        df = self._onehot_encoding(df, cat_cols)
        for col in num_cols + level_col:
            df[col] = df[col].apply(lambda x: np.log(x + 1))

        return df

    def _split_train_test(self, df):
        """This function generates train and test sets"""
        self.y = df[self.label_col].values
        self.X = df[[col for col in df.columns if col != self.label_col]]
        X = df.drop(self.label_col, axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, self.y, test_size=0.10, random_state=10, stratify=self.y
        )

        sm = SMOTE(random_state=12)
        X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
        # print("X_train", X_train.shape)
        # print("X_test", X_test.shape)
        self.feature_train = X_train_sm
        self.target_train = y_train_sm
        self.feature_test = X_test
        self.target_test = y_test
        self.scaler = scaler

    def preprocess_newdata(self, nparray):
        columns = [
            "male",
            "age",
            "education",
            "currentSmoker",
            "cigsPerDay",
            "BPMeds",
            "prevalentStroke",
            "prevalentHyp",
            "diabetes",
            "totChol",
            "sysBP",
            "diaBP",
            "BMI",
            "heartRate",
            "glucose",
        ]
        df = pd.DataFrame(nparray, columns=columns)
        df = self._onehot_newdata(df)
        for col in self.num_cols + self.level_cols:
            df[col] = df[col].apply(lambda x: np.log(x + 1))
        X = self.scaler.transform(df)
        return X


# test function

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    # define input files
    data_file = "/home/giangdip/Giangdip/HUS/Python/Project_Final/framingham.csv"
    numeric_var = [
        "age",
        "cigsPerDay",
        "totChol",
        "sysBP",
        "diaBP",
        "BMI",
        "heartRate",
        "glucose",
    ]
    level_var = ["education"]
    category_var = [
        "male",
        "currentSmoker",
        "BPMeds",
        "prevalentStroke",
        "prevalentHyp",
        "diabetes",
    ]
    target = ["TenYearCHD"]

    # Create Data object
    data = Datasets(
        data_file=data_file,
        cat_cols=category_var,
        num_cols=numeric_var,
        level_cols=level_var,
        label_col=target,
        train=True,
    )

    X = data.feature_train
    y = data.target_train
    clf = linear_model.LogisticRegression()
    clf2 = RandomForestClassifier(
        criterion="gini",
        n_estimators=1750,
        max_depth=7,
        min_samples_split=6,
        min_samples_leaf=6,
        max_features="auto",
        oob_score=True,
        random_state=10,
        n_jobs=-1,
        verbose=1,
    )
    clf2.fit(X, y)
    preds = clf2.predict(data.feature_test)
    print(accuracy_score(preds, data.target_test))
    print(roc_auc_score(preds, data.target_test))
