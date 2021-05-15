import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from datasets import Datasets
from skopt.learning import GaussianProcessRegressor
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import joblib
from sklearn import metrics
from sklearn import svm
from skopt import BayesSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin, space_eval, tpe
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import autosklearn.classification
from sklearn.neural_network import MLPClassifier
from functools import partial


# squared exponential kernel
from sklearn.gaussian_process.kernels import RBF
from skopt.utils import use_named_args

# https://github.com/scikit-optimize/scikit-optimize/issues/978
def bayes_search_CV_init(
    self,
    estimator,
    search_spaces,
    optimizer_kwargs=None,
    n_iter=50,
    scoring=None,
    fit_params=None,
    n_jobs=1,
    n_points=1,
    iid=True,
    refit=True,
    cv=None,
    verbose=0,
    pre_dispatch="2*n_jobs",
    random_state=None,
    error_score="raise",
    return_train_score=False,
):

    self.search_spaces = search_spaces
    self.n_iter = n_iter
    self.n_points = n_points
    self.random_state = random_state
    self.optimizer_kwargs = optimizer_kwargs
    self._check_search_space(self.search_spaces)
    self.fit_params = fit_params

    super(BayesSearchCV, self).__init__(
        estimator=estimator,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit,
        cv=cv,
        verbose=verbose,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
        return_train_score=return_train_score,
    )


BayesSearchCV.__init__ = bayes_search_CV_init


class Model:

    """initializes models, dicts"""

    def __init__(self, df):
        self.df = df
        self.best_score = 0
        self.run = self.train(df)
        self.best_accuracy
        self.best_model
        self.best_model_name
        self.best_param

    def add_model(self, model):
        """Adds models to the list"""
        self.models.append(model)

    def _GradientBoosting(self, df):
        print("Start GradientBoosting...")
        param_grid = [
            Integer(10, 120, name="n_estimators"),
            Real(0, 0.999, name="min_samples_split"),
            Integer(1, 5, name="max_depth"),
            Categorical(["deviance", "exponential"], name="loss"),
        ]

        # set up the gradient boosting classifier

        gbm = GradientBoostingClassifier(random_state=0)
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

        gpr = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, noise="gaussian", n_restarts_optimizer=2
        )
        # the decorator allows our objective function to receive the parameters as

        @use_named_args(param_grid)
        def objective(**params):

            # model with new parameters
            gbm.set_params(**params)

            # optimization function (hyperparam response function)
            value = np.mean(
                cross_val_score(
                    gbm,
                    df.feature_train,
                    df.target_train,
                    cv=5,
                    n_jobs=-4,
                    scoring="roc_auc",
                )  # "accuracy"
            )

            # negate because we need to minimize
            return -value

        gp_ = gp_minimize(
            objective,
            dimensions=param_grid,
            base_estimator=gpr,
            n_initial_points=5,
            acq_optimizer="sampling",
            random_state=42,
        )
        params = {
            "n_estimators": gp_.x[0],
            "min_samples_split": gp_.x[1],
            "max_depth": gp_.x[2],
            "loss": gp_.x[3],
        }
        final_model = GradientBoostingClassifier(
            n_estimators=gp_.x[0],
            min_samples_split=gp_.x[1],
            max_depth=gp_.x[2],
            loss=gp_.x[3],
        )
        final_model.fit(df.feature_train, df.target_train)
        y_preds = final_model.predict(df.feature_test)
        joblib.dump(final_model, "./log/GradientBoosting.pkl")
        return {
            "GradientBoosting": {
                "Best_score": metrics.roc_auc_score(y_preds, df.target_test),
                "Best_accuracy": metrics.accuracy_score(y_preds, df.target_test),
                "params": params,
                "model": final_model,
            }
        }

    def _Losgistic(self, df):
        print("Start Logistic...")
        param_grid = [
            Categorical(["newton-cg", "lbfgs", "liblinear"], name="solver"),
            Categorical(["l2"], name="penalty"),
            Real(1e-5, 100, name="C"),
        ]

        # set up the logistic regressoion classifier

        lgt = LogisticRegression(random_state=0)
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

        gpr = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, noise="gaussian", n_restarts_optimizer=2
        )
        # the decorator allows our objective function to receive the parameters as

        @use_named_args(param_grid)
        def objective(**params):

            # model with new parameters
            lgt.set_params(**params)

            # optimization function (hyperparam response function)
            value = np.mean(
                cross_val_score(
                    lgt,
                    df.feature_train,
                    df.target_train,
                    cv=5,
                    n_jobs=-4,
                    scoring="roc_auc",
                )  # "accuracy"
            )

            # negate because we need to minimize
            return -value

        gp_ = gp_minimize(
            objective,
            dimensions=param_grid,
            base_estimator=gpr,
            n_initial_points=5,
            acq_optimizer="sampling",
            random_state=42,
        )
        params = {
            "solver": gp_.x[0],
            "penalty": gp_.x[1],
            "C": gp_.x[2],
        }
        final_model = LogisticRegression(
            solver=gp_.x[0],
            penalty=gp_.x[1],
            C=gp_.x[2],
        )
        final_model.fit(df.feature_train, df.target_train)
        y_preds = final_model.predict(df.feature_test)
        joblib.dump(final_model, "./log/Logistic.pkl")
        return {
            "Logistic": {
                "Best_score": metrics.roc_auc_score(y_preds, df.target_test),
                "Best_accuracy": metrics.accuracy_score(y_preds, df.target_test),
                "params": params,
                "model": final_model,
            }
        }

    def _RandomForest(self, df):
        print("Start RandomForest...")
        param_grid = [
            Categorical(["sqrt", "log2", None], name="max_features"),
            Integer(120, 1200, name="n_estimators"),
            Integer(5, 30, name="max_depth"),
            Integer(2, 15, name="min_samples_split"),
            Integer(1, 10, name="min_samples_leaf"),
        ]

        # set up the logistic regressoion classifier

        rf = RandomForestClassifier(random_state=0)
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

        gpr = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, noise="gaussian", n_restarts_optimizer=2
        )
        # the decorator allows our objective function to receive the parameters as

        @use_named_args(param_grid)
        def objective(**params):

            # model with new parameters
            rf.set_params(**params)

            # optimization function (hyperparam response function)
            value = np.mean(
                cross_val_score(
                    rf,
                    df.feature_train,
                    df.target_train,
                    cv=5,
                    n_jobs=-4,
                    scoring="roc_auc",
                )  # "accuracy"
            )

            # negate because we need to minimize
            return -value

        gp_ = gp_minimize(
            objective,
            dimensions=param_grid,
            base_estimator=gpr,
            n_initial_points=5,
            acq_optimizer="sampling",
            random_state=42,
        )
        params = {
            "max_features": gp_.x[0],
            "n_estimators": gp_.x[1],
            "max_depth": gp_.x[2],
            "min_samples_split": gp_.x[3],
            "min_samples_leaf": gp_.x[4],
        }
        final_model = RandomForestClassifier(
            max_features=gp_.x[0],
            n_estimators=gp_.x[1],
            max_depth=gp_.x[2],
            min_samples_split=gp_.x[3],
            min_samples_leaf=gp_.x[4],
        )
        final_model.fit(df.feature_train, df.target_train)
        y_preds = final_model.predict(df.feature_test)
        joblib.dump(final_model, "./log/RandomForest.pkl")
        return {
            "RandomForest": {
                "Best_score": metrics.roc_auc_score(y_preds, df.target_test),
                "Best_accuracy": metrics.accuracy_score(y_preds, df.target_test),
                "params": params,
                "model": final_model,
            }
        }

    def _SuportVector(self, df):
        """Suport vector machine using grid search"""
        print("Start SuportVector...")
        # defining parameter range
        param_grid = {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": ["auto"],
            "kernel": ["rbf", "linear", "poly"],
            "class_weight": ["balanced", None],
        }
        cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
        sv = GridSearchCV(
            SVC(),
            param_grid,
            scoring="roc_auc",
            cv=cross_validation,
        )
        sv.fit(df.feature_train, df.target_train)
        y_preds = sv.predict(df.feature_test)
        joblib.dump(sv, "./log/SuportVector.pkl")
        return {
            "SuportVector": {
                "Best_score": metrics.roc_auc_score(y_preds, df.target_test),
                "Best_accuracy": metrics.accuracy_score(y_preds, df.target_test),
                "params": sv.get_params(),
                "model": sv,
            }
        }

    def _XGBoost(self, df):
        """Tuning hyper paramater of XGBoost using Hyperot"""
        print("Start XGBoost...")

        def objective(space):
            classifier = XGBClassifier(
                n_estimators=space["n_estimators"],
                max_depth=int(space["max_depth"]),
                learning_rate=space["learning_rate"],
                gamma=space["gamma"],
                min_child_weight=space["min_child_weight"],
                subsample=space["subsample"],
                colsample_bytree=space["colsample_bytree"],
            )

            classifier.fit(df.feature_train, df.target_train)
            # Applying k-Fold Cross Validation

            accuracies = cross_val_score(
                estimator=classifier, X=df.feature_train, y=df.target_train, cv=5
            )
            CrossValMean = accuracies.mean()

            return {"loss": 1 - CrossValMean, "status": STATUS_OK}

        # space of parameters
        space = {
            "max_depth": hp.choice("max_depth", range(5, 30, 1)),
            "learning_rate": hp.quniform("learning_rate", 0.01, 0.5, 0.01),
            "n_estimators": hp.choice("n_estimators", range(20, 205, 5)),
            "gamma": hp.quniform("gamma", 0, 0.50, 0.01),
            "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
            "subsample": hp.quniform("subsample", 0.1, 1, 0.01),
            "colsample_bytree": hp.quniform("colsample_bytree", 0.1, 1.0, 0.01),
        }
        # find the best
        trials = Trials()
        best = fmin(
            fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials
        )
        xbg = XGBClassifier(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            learning_rate=best["learning_rate"],
            gamma=best["gamma"],
            min_child_weight=best["min_child_weight"],
            subsample=best["subsample"],
            colsample_bytree=best["colsample_bytree"],
        )
        xbg.fit(df.feature_train, df.target_train)
        y_preds = xbg.predict(df.feature_test)
        joblib.dump(xbg, "./log/XGBoost.pkl")
        return {
            "XGBoost": {
                "Best_score": metrics.roc_auc_score(y_preds, df.target_test),
                "Best_accuracy": metrics.accuracy_score(y_preds, df.target_test),
                "params": best,
                "model": xbg,
            }
        }

    def _NeuralNetwork(self, df):
        print("Start NeuralNetwork...")

        def objective(params, model, x, y, k=5):
            model.set_params(**params)
            kfold = StratifiedKFold(n_splits=k)
            score = cross_val_score(
                model, df.feature_train, df.target_train, cv=kfold, scoring="accuracy"
            )
            return 1 - score.mean()

        def optimize_params(model, x, y, space, k=5, max_evals=100, eval_space=False):
            trials = Trials()
            best = fmin(
                partial(objective, model=model, x=x, y=y, k=k),
                space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
            )

            param_values = [t["misc"]["vals"] for t in trials.trials]
            param_values = [
                {key: value for key in params for value in params[key]}
                for params in param_values
            ]

            if eval_space:
                param_values = [space_eval(space, params) for params in param_values]

            param_df = pd.DataFrame(param_values)
            param_df["accuracy"] = [1 - loss for loss in trials.losses()]
            return space_eval(space, best), param_df

        space_mlp = {}
        space_mlp["hidden_layer_sizes"] = 10 + hp.randint("hidden_layer_sizes", 40)
        space_mlp["alpha"] = hp.loguniform("alpha", -8 * np.log(10), 3 * np.log(10))
        space_mlp["activation"] = hp.choice("activation", ["relu", "logistic", "tanh"])
        space_mlp["solver"] = hp.choice("solver", ["lbfgs", "sgd", "adam"])
        model = MLPClassifier()
        best_mlp, param_mlp = optimize_params(
            model, df.feature_train, df.target_train, space_mlp, max_evals=100
        )
        mlp = MLPClassifier(
            hidden_layer_sizes=best_mlp["hidden_layer_sizes"],
            alpha=best_mlp["alpha"],
            activation=best_mlp["activation"],
            solver=best_mlp["solver"],
        )
        mlp.fit(df.feature_train, df.target_train)
        y_preds = mlp.predict(df.feature_test)
        joblib.dump(mlp, "./log/NeuralNetwork.pkl")
        return {
            "NeuralNetwork": {
                "Best_score": metrics.roc_auc_score(y_preds, df.target_test),
                "Best_accuracy": metrics.accuracy_score(y_preds, df.target_test),
                "params": best_mlp,
                "model": mlp,
            }
        }

    def _autoSK(self, df):
        print("Start AutoSK...")
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=120,
            per_run_time_limit=30,
            disable_evaluator_output=False,
            resampling_strategy="cv",
            resampling_strategy_arguments={"folds": 5},
        )
        automl.fit(df.feature_train, df.target_train)
        y_preds = automl.predict(df.feature_test)
        joblib.dump(automl, "./log/AutoSK.pkl")
        return {
            "AutoSK": {
                "Best_score": metrics.roc_auc_score(y_preds, df.target_test),
                "Best_accuracy": metrics.accuracy_score(y_preds, df.target_test),
                "params": None,
                "model": automl,
            }
        }

    def _select_model(self, name, result):
        sample_auc_score = result[name]["Best_score"]
        sample_acc_score = result[name]["Best_accuracy"]
        sample_parms = result[name]["params"]
        print("Start Logging...")
        with open(os.path.join("log", f"log_{name}.txt"), "w") as f:
            f.writelines(f"AUC score : {str(sample_auc_score)}")
            f.writelines("\n")
            f.writelines(f"Accuracy score : {str(sample_acc_score)}")
            f.writelines("\n")
            f.writelines(f"params : {str(sample_parms)}")
        f.close()
        if result[name]["Best_score"] > self.best_score:
            self.best_score = result[name]["Best_score"]
            self.best_accuracy = result[name]["Best_accuracy"]
            self.best_model = result[name]["model"]
            self.best_model_name = name
            self.best_param = result[name]["params"]

    def train(self, df):
        """prints summary of models, best model"""

        model_list = [
            "AutoSK",
            "NeuralNetwork",
            "XGBoost",
            "SuportVector",
            "RandomForest",
            "Logistic",
            "GradientBoosting",
        ]
        for name in model_list:
            file = name + ".pkl"
            if not os.path.isfile(os.path.join("./log", file)):
                if name == "AutoSK":
                    self._select_model(name, self._autoSK(df))
                elif name == "NeuralNetwork":
                    self._select_model(name, self._NeuralNetwork(df))
                elif name == "XGBoost":
                    self._select_model(name, self._XGBoost(df))
                elif name == "SuportVector":
                    self._select_model(name, self._SuportVector(df))
                elif name == "RandomForest":
                    self._select_model(name, self._RandomForest(df))
                elif name == "Logistic":
                    self._select_model(name, self._Losgistic(df))
                elif name == "GradientBoosting":
                    self._select_model(name, self._GradientBoosting(df))
                else:
                    print("Model is not support yet")
            else:
                model_ = joblib.load(os.path.join("./log", file))
                y_preds = model_.predict(df.feature_test)
                model_score = (metrics.roc_auc_score(y_preds, df.target_test),)
                model_accuracy = metrics.accuracy_score(y_preds, df.target_test)
                if model_score > self.best_score:
                    self.best_score = model_score
                    self.best_accuracy = model_accuracy
                    self.best_model = model_
                    self.best_model_name = name
                    self.best_param = None
        print(f"Best model is : {self.best_model_name}")
        print(f"Best AUC score is : {self.best_score}")
        print(f"Best accuracy score is : {self.best_accuracy}")
        print(f"Best params are : {self.best_param}")
        print("Saving...best model")
        joblib.dump(self.best_model, f"./log/best_model/{self.best_model_name}.pkl")


if __name__ == "__main__":
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

    X_train = data.feature_train
    y_train = data.target_train

    X_test = data.feature_test
    y_test = data.feature_test
    # training data
    models = Model(data)
