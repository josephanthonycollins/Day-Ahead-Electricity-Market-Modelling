# -*- coding: utf-8 -*-
"""
Created on 28th December 2024.

@author: Joseph Collins, Part-time Phd Student, University College Cork, Department of Applied Mathematics, email 98718584@umail.ucc.ie, Supervisors = {Dr. Andreas Amann, Dr. Kieran Mulchrone}

Purpose and Context: This file is sample/mock code, it contains helper functions which will be used in the sample_code.py module.
                     Note, this file is for demo purposes only, it is not anticipated to be run in practice (e.g. assumes access to a mysql database to persist results etc).
"""

import os
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from datetime import timedelta as td
import traceback
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from tensorflow.keras import models
from keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from keras.initializers import he_uniform
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


def multitarget_walk_forward_validation_wrapper(participant_df: pd.DataFrame, db_table_name: str, unit_names: [str], raw_csv_location: str, model_dictionary: dict, targets: [str], explanatory_variables: [str],
                                                interaction_terms: [str], features_to_encode: [str], meta_settings: [dict], model_settings: [dict], start_time: dt, end_time: dt,
                                                naive_estimator_columns=None, dataset_type="_basic_ts_dataset", save_summary_to_db=False, save_test_to_csv=False, csv_output_location=None, plot=False,
                                                clock_change_start=None, clock_change_end=None, min_number_of_calibration_points=None,earlystopping=None, test_output=None):
    """
    This method is used to wrap the whole process. It can be used to run a single, or multiple scenarios.
    Each scenario does a rolling calibration and prediction for each day in the time horizon (i.e. walk forward validation), the output is then saved to the database (and output to a csv).
    The method can handle a number of different scenarios and meta parameters.

    Parameters
    ----------
    db_table_name : str
        The database table name that is to save the output.
    unit_names : [str]
        The participants that we want to fit the scenarios/models to.
    raw_csv_locaiton : str
        Location of the input csv file for the participant of interest.
    model_dictionary : dict
        The dictionary indicating which model type we are interested in fitting.
    targets : [str]
        The variables that we are interested in forecasting/predicting.
    explanatory_variables : [str]
        The explanatory variables, will be used in the model to try and predict/forecast the targets.
    interaction_terms : [str]
        If we want to include any interaction terms as explanatory variables, we list them here.
    features_to_encode : [str]
        If there are any variables we want to apply one hot encoding to, we list them here.
    meta_settings : [dict]
        List of dictionaries, each dictionary contains a meta scenario that we want to apply.
    model_settings : [dict]
        List of dictionaries, each dictionary contains a model setting that we want to apply.
    start_time : dt
        A rolling approach will be applied to data between start_time and end_time. This consists of calibrating to data in [start_time + look back days, start] and then predicting output for [start time, start time + 1 day].
    end_time : dt
        See previous comment.
    naive_estimator_columns: [str]
        These are the strings that will be used to predict i.e. the lagged variable will be used to produce a forecast/prediciton.
    dataset_type: str
        There are two input dataset types for each participant. This variable just indicates which one will be used as input.
    save_summary_to_dbv: boolean
        Flag used to indicate whether or not, summary info, should be saved to the database.
    save_test_to_csv: boolean
        Flag used to indicate whether or not, output, should be saved to csv.
    csv_output_location: str
        If save_test_to_csv is true, then the csv will be saved to the location specified by csv_output_locaiton.
    plot: boolean
        Whether or not to plotthe cv test set forecast versus actuals.
    clock_change_start: [datetime]
        List of datetimes relating to daylight savings. If provided, then forecasts will not be produced for delivery hours in the interval [clock_change_start[i],clock_change_end[i]].
    clock_change_end: [datetime]
        See previous. comment.
    min_number_of_calibration_points: int
        Integer specifying the minimum number of data points needed in order to calibrate the model and produce a forecast.
    earlystopping: str
        If not none, then for a neural network, earlystopping applies.

    Returns
    -------
    Tuple of string lists, error_messages and info_messages.
    For each iteration, if there are any errors, a related message will be saved in error_messages.
    Similarly, for info_messages.

    """
    error_messages = []
    info_messages = []
    results_dataframe = None

    try:

        print("helper_functions, multitarget_walk_forward_validation_wrapper, start.")

        ffnn = model_dictionary.get("ffnn", False)
        rf = model_dictionary.get("rf", False)
        svm = model_dictionary.get("svm", False)
        knn = model_dictionary.get("knn", False)
        lasso = model_dictionary.get("lasso",False)
        elasticnet = model_dictionary.get("elasticnet",False)
        xgb = model_dictionary.get("xgb",False)
        naive_estimator_2day_lookback = model_dictionary.get("naive_estimator_2day_lookback", False)
        naive_estimator_7day_lookback = model_dictionary.get("naive_estimator_7day_lookback", False)

        # Get the unit type information from the database.
        for i in range(0, len(unit_names)):

            iteration_counter = 0

            unit_name = unit_names[i]

            for j in range(0, len(meta_settings)):

                for k in range(0, len(model_settings)):

                    try:

                        results = []
                        df_columns = []

                        meta_dictionary = meta_settings[j]
                        model_dictionary = model_settings[k]

                        # Check if that particular experiment/setting has been run previously.
                        scenario_previously_run = None
                        explanatory_variables = meta_dictionary.get("explanatory_variables", explanatory_variables)
                        features_to_encode = meta_dictionary.get("features_to_encode", features_to_encode)
                        model = None
                        if ffnn:
                            model = "ffnn"
                        if svm:
                            model = "svm"
                        if rf:
                            model = "rf"
                        if knn:
                            model = "knn"
                        if lasso:
                            model = "lasso"
                        if elasticnet:
                            model = "elasticnet"
                        if xgb:
                            model = "xgb"
                        if naive_estimator_2day_lookback:
                            model = "naive_estimator_2day_lookback"
                        if naive_estimator_7day_lookback:
                            model = "naive_estimator_7day_lookback"

                        scenario_previously_run = False

                        unit_df_ts_str = unit_name + dataset_type

                        participant_df = participant_df.dropna()

                        if len(participant_df) == 0:
                            print("No data for " + unit_name + ", moving onto next iteration...")
                            info_messages.append("No data for " + unit_name + ".")
                            continue

                        run_time = datetime.now()

                        train_mae_dictionary = None
                        test_output_df = None
                        error_m = None
                        info_m = None

                        # Run the walk forward validation and save the csv output (actual and forecast).
                        train_mae_dictionary, test_output_df, error_m, info_m = multitarget_walk_forward_validation(participant_df=participant_df, targets=targets, explanatory_variables=explanatory_variables, interaction_terms=interaction_terms, features_to_encode=features_to_encode,
                                                                                                                    meta_dictionary=meta_dictionary, model_dictionary=model_dictionary, unit_name=unit_name, start_time=start_time, end_time=end_time, rf=rf, knn=knn,
                                                                                                                    ffnn=ffnn, svm=svm, lasso=lasso, xgb=xgb, elasticnet=elasticnet, naive_estimator_2day_lookback=naive_estimator_2day_lookback, naive_estimator_7day_lookback=naive_estimator_7day_lookback,
                                                                                                                    naive_estimator_columns=naive_estimator_columns, save_test_to_csv=save_test_to_csv, csv_output_location=csv_output_location, csv_name=db_table_name, plot=plot,
                                                                                                                    clock_change_start=clock_change_start, clock_change_end=clock_change_end, min_number_of_calibration_points=min_number_of_calibration_points,earlystopping=earlystopping,
                                                                                                                    test_output=test_output)

                        if len(error_m) > 0:
                            for k in range(0, len(error_m)):
                                error_messages.append(unit_name + " " + error_m[k])

                        if len(info_m) > 0:
                            for k in range(0, len(info_m)):
                                info_messages.append(unit_name + " " + info_m[k])

                        df_columns.append("Run Time")
                        results.append(run_time)

                        df_columns.append("Test_Start")
                        results.append(start_time)

                        df_columns.append("Test_End")
                        results.append(end_time)

                        df_columns.append("Unit")
                        results.append(unit_name)

                        df_columns.append("Targets")
                        results.append(str(targets))

                        df_columns.append("Explanatory")
                        results.append(str(explanatory_variables))

                        df_columns.append("Interaction")
                        results.append(str(interaction_terms))

                        df_columns.append("Categorical")
                        results.append(str(features_to_encode))

                        df_columns.append("Scale_Explanatory")
                        results.append(meta_dictionary.get(
                            "scale_explanatory_variables"))

                        df_columns.append("Scale_Target")
                        results.append(meta_dictionary.get(
                            "apply_target_scaling"))

                        df_columns.append("Calibration")
                        results.append(meta_dictionary.get(
                            "calibration_lookback_days"))

                        df_columns.append("Model")
                        df_columns.append("Settings")
                        if ffnn:
                            results.append("ffnn")
                            results.append(str(model_dictionary))
                        if svm:
                            results.append("svm")
                            results.append(str(model_dictionary))
                        if rf:
                            results.append("rf")
                            results.append(str(model_dictionary))
                        if knn:
                            results.append("knn")
                            results.append(str(model_dictionary))
                        if lasso:
                            results.append("lasso")
                            results.append(str(model_dictionary))
                        if elasticnet:
                            results.append("elasticnet")
                            results.append(str(model_dictionary))
                        if xgb:
                            results.append("xgb")
                            results.append(str(model_dictionary))
                        if naive_estimator_2day_lookback:
                            results.append("naive_estimator_2day_lookback")
                            results.append(str(naive_estimator_columns))
                        if naive_estimator_7day_lookback:
                            results.append("naive_estimator_7day_lookback")
                            results.append(str(naive_estimator_columns))

                        df_columns.append("test_days")
                        temp = train_mae_dictionary.get(targets[0], None)
                        if temp is not None:
                            results.append(len(temp))
                        else:
                            results.append(None)

                        df_columns.append("flagged_test_days")
                        temp = train_mae_dictionary.get("Flagged", None)
                        if temp is not None:
                            results.append(temp.count(True))
                        else:
                            results.append(None)

                        if len(meta_settings) > 1 or len(model_settings) > 1:
                            df_columns.append("iteration_counter")
                            results.append(iteration_counter)
                            iteration_counter += 1

                        for target in targets:
                            if test_output_df.empty:
                                df_columns.append(target+"_train_avg_mae")
                                df_columns.append(target+"_test_mae")
                                df_columns.append(target+"_test_rmae")
                                results.append(None)
                                results.append(None)
                                results.append(None)
                            else:
                                df_columns.append(target+"_train_avg_mae")
                                train_mae_values = None
                                train_mae_values = train_mae_dictionary.get(target, None)
                                if train_mae_values is not None:
                                    cv_train_avg_mae = None
                                    if len(train_mae_values) > 0:
                                      cv_train_avg_mae = round(sum(train_mae_values)/len(train_mae_values), 2)
                                    results.append(cv_train_avg_mae)
                                df_columns.append(target+"_test_mae")
                                df_columns.append(target+"_test_rmae")
                                predictor_test_mae = None
                                predictor_test_rmae = None
                                predictor_test_mae = round(mean_absolute_error(test_output_df[target], test_output_df[target+"_Forecast"]), 2)
                                predictor_test_rmae = None
                                try:
                                    predictor_test_naive_mae = round(mean_absolute_error(test_output_df[target], test_output_df[target+"_-48"]), 2)
                                    predictor_test_rmae = round(predictor_test_mae / predictor_test_naive_mae, 2)
                                    if math.isinf(predictor_test_rmae):
                                        raise Exception(target + " rmae is inf")
                                except Exception:
                                    predictor_test_rmae = None
                                results.append(predictor_test_mae)
                                results.append(predictor_test_rmae)

                        temp = []
                        temp.append(results)
                        results_dataframe = pd.DataFrame(temp, columns=df_columns)

                        return results_dataframe, error_messages, info_messages

                    except Exception:

                        print("Error: helper_functions, multitarget_walk_forward_validation_wrapper. Moving onto next iteration")
                        traceback.print_exc()
                        error_messages.append("Error occured for participant " + unit_name)
                        continue

        print("helper_functions, multitarget_walk_forward_validation_wrapper, end.")

        return results_dataframe, error_messages, info_messages

    except Exception:
        print("Error: helper_functions, multitarget_walk_forward_validation_wrapper.")
        traceback.print_exc()
        return results_dataframe, error_messages, info_messages


def multitarget_walk_forward_validation(participant_df: pd.DataFrame, targets: [str], explanatory_variables: [str], interaction_terms: [str], features_to_encode: [str], meta_dictionary: dict, model_dictionary: dict,
                                        unit_name: str, start_time: dt, end_time: dt, rf=False, knn=False, ffnn=False, svm=False, lasso=False, xgb=False, elasticnet=False, naive_estimator_2day_lookback=False, naive_estimator_7day_lookback=False, naive_estimator_columns=None,
                                        save_test_to_csv=False, csv_output_location=None, csv_name=None, plot=False, clock_change_start=None, clock_change_end=None, min_number_of_calibration_points=148,earlystopping=None,
                                        test_output=None):
    """
    This method is used to do a walk forward validation on a participant dataset.
    It returns the forecast versus actuals.
    It also returns a dictionary containing mae for each of the target variables during training.

    Parameters
    ----------
    participant_df : pd.DataFrame
        The pandas dataframe containing all the variables of interest.
    targets : [str]
        The variables, or targets, that we are interested in forecasting/predicting.
    explanatory_variables : [str]
        The explanatory variables, will be used in the model to try and predict/forecast the targets.
    interaction_terms : [str]
        If we want to include any interaction terms as explanatory variables, we list them here.
    features_to_encode : [str]
        If there are any variables we want to apply one hot encoding to, we list them here.
    meta_dictionary : dict
        Settings that we want to apply to the data/modes.
    model_dictionary : dict
        Model specific settings.
    unit_name: str
        String that will be used in the naming convention (for csv output if it is used, or graphs if it is used).
    start_time : dt
        A rolling approach will be applied to data between start_time and end_time. This consists of calibrating to data in [start_time + look back days, start] and then predicting output for [start time, start time + 1 day].
    end_time : dt
        See previous comment.
    rf: boolean
        If we are applying a random forest model.
    knn/ffnn/svm/naive_estimator_2day_lookback/naive_estimator_7day_lookback: boolean
        Similar to the previous comment, except specifying whether it is K nearest neighbours, feed forward neural network, support vector machine or naive.
    naive_estimator_columns: [str]
        These are the strings that will be used to predict i.e. the lagged variable will be used to produce a forecast/prediciton.
    save_test_to_csv: boolean
        Flag used to indicate whether or not, (b) reference below, should be saved to csv.
    csv_output_location: str
        If save_test_to_csv is true, then the csv will be saved to the location specified by csv_output_locaiton.
    csv_name: str
        If save_test_to_csv is true, then the csv naming convention will be model + " " + unit_name + " " + csv_name.
    plot: boolean
        Whether or not to plotthe cv test set forecast versus actuals.
    clock_change_start: [datetime]
        List of datetimes relating to daylight savings. If provided, then forecasts will not be produced for delivery hours in the interval [clock_change_start[i],clock_change_end[i]].
    clock_change_end: [datetime]
        See previous. comment.
    min_number_of_calibration_points: int
        The minimum number of observations that should exist in order to perform the calibration, otherwise a naive or zero estiamte will be used.
    earlystopping: str
        If it is not none, then earlystopping will apply to the neural network.

    Returns
    -------
    A tuple containing
        a) dictionary containing the mae for each target for each iteration in the walk forward validation.
        b) pd.DataFrame containing the forecast and actuals for each iteration in the walk forward validation.
        c) [str] containing error_messages for each iteration in the walk forward validation.
        d) [str] containing info messages for each iteration in the walk forward validation.

    """
    error_messages = []
    info_messages = []
    try:

        print("helper_functions, multitarget_walk_forward_validation, start.")

        date_time_column = "DeliveryPeriod"

        train_mae_results = dict()
        train_mae_results["Flagged"] = []
        train_mae_results["Reason"] = []
        for target in targets:
            train_mae_results[target] = []
        test_forecast_and_actuals = None
        if test_output is not None and len(test_output) > 0:
            test_forecast_and_actuals = test_output.copy()
            last_forecast_time = test_forecast_and_actuals["DeliveryPeriod"].max()
            start_time = dt.datetime(last_forecast_time.year,last_forecast_time.month,last_forecast_time.day,23,0,0)
        else:
            test_forecast_and_actuals = pd.DataFrame()

        scale_explanatory_variables = meta_dictionary.get('scale_explanatory_variables', None)
        apply_target_scaling = meta_dictionary.get('apply_target_scaling', None)
        calibration_lookback_days = meta_dictionary.get('calibration_lookback_days', None)

        all_columns = list(participant_df.columns)
        if len(features_to_encode) > 0:
            columns_to_exclude = ['index', 'level_0', 'DeliveryPeriod'] + features_to_encode + targets
            prefix_to_include = []
            for feature in features_to_encode:
                prefix_to_include.append(feature[0]+"_")
        else:
            columns_to_exclude = [
                item for item in all_columns if item not in explanatory_variables]
            prefix_to_include = []

        model = None
        model_str = None

        if rf:
            n_estimators = model_dictionary.get('n_estimators', 100)
            max_depth = model_dictionary.get('max_depth', 16)
            max_features = model_dictionary.get('max_features', 20)
            min_samples_leaf = model_dictionary.get('min_samples_leaf', 5)
            criterion = model_dictionary.get('criterion', "squared_error")
            model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, n_jobs=-1, oob_score=False, bootstrap=False,
                                          random_state=42, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf)
            model_str = "RF " + unit_name + " " + csv_name

        if knn:
            n_neighbors = model_dictionary.get('n_neighbors', 100)
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
            model_str = "KNN " + unit_name + " " + csv_name

        if svm:
            C = model_dictionary.get("C",56)
            epsilon = model_dictionary.get("epsilon",1.5)
            svm_rbf_reg = SVR(kernel="rbf", C=C, epsilon=epsilon)
            model = MultiOutputRegressor(svm_rbf_reg)
            model_str = "SVM " + unit_name + " " + csv_name

        if lasso:
            alpha = model_dictionary.get('alpha', 0.01)
            fit_intercept = model_dictionary.get('fit_intercept', False)
            normalize = model_dictionary.get('normalize', False)
            model = Lasso(alpha=alpha,fit_intercept=fit_intercept, normalize=normalize, precompute=False, copy_X=True, max_iter=2000, tol=0.01, warm_start=False, positive=False, random_state=42)
            model_str = "Lasso " + unit_name + " " + csv_name

        if elasticnet:
            alpha = model_dictionary.get('alpha', 1.0)
            l1_ratio = model_dictionary.get('l1_ratio', 0.5)
            fit_intercept = model_dictionary.get('fit_intercept', False)
            normalize = model_dictionary.get('normalize', False)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, normalize=normalize, precompute=False, copy_X=True, max_iter=2000, tol=0.01, warm_start=False, positive=False, random_state=42)
            model_str = "ElasticNet " + unit_name + " " + csv_name

        if xgb:
            n_estimators = model_dictionary.get('n_estimators', 100)
            max_depth = model_dictionary.get('max_depth', 3)
            learning_rate = model_dictionary.get('learning_rate', 0.1)
            subsample = model_dictionary.get('subsample', 0.8)
            reg_alpha = model_dictionary.get('reg_alpha', 0.1)
            model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, colsample_bytree=subsample, reg_alpha=reg_alpha)
            model_str = "XGB " + unit_name + " " + csv_name

        if naive_estimator_2day_lookback:
            model_str = "Naive 2Day Lookback" + unit_name + " " + csv_name

        if naive_estimator_7day_lookback:
            model_str = "Naive 7Day Lookback" + unit_name + " " + csv_name

        global_m_epochs = None
        m_epochs = None
        m_verbose = None
        need_to_concat = None
        drop_out = None
        learning_rate = None

        if ffnn:
            number_of_nodes_list = model_dictionary.get('number_of_nodes_list', [32, 32, 32])
            global_m_epochs = model_dictionary.get('m_epochs', 100)
            m_verbose = 0
            need_to_concat = True
            keep_epochs_unchanged = model_dictionary.get('keep_epochs_unchanged', False)
            learning_rate = model_dictionary.get('learning_rate', 0.001)
            drop_out = model_dictionary.get('drop_out', 0.2)
            model_str = "FFNN " + unit_name + " " + csv_name

        iteration = 0

        # We do a rolling
        #   (a) train
        #   (b) forecast on 1 day ahead day ahead data.

        while start_time < end_time and (rf or knn or naive_estimator_2day_lookback or naive_estimator_7day_lookback or svm or ffnn or lasso or elasticnet or xgb):

            # Check that we aren't in a daylight savings day as we don't want to produce a forecast for that day
            skip = False
            for i in range(0, len(clock_change_start)):
                if (clock_change_start[i] <= start_time) and (clock_change_end[i] >= start_time):
                    skip = True
                    break

            if skip == True:
                message = "Daylight savings, " + str(start_time) + ", skipping."
                print(message)
                info_messages.append(message)
                start_time = start_time + td(days=1)
                continue

            # Calibration interval
            calibration_start_time = start_time + td(days=calibration_lookback_days)
            calibration_end_time = start_time

            # Test interval, the test period is always the day ahead forecast
            test_start_time = calibration_end_time
            test_end_time = test_start_time + td(days=1)

            print("calibration_start_time: " + str(calibration_start_time) + ", calibration_end_time: " + str(calibration_end_time) +
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))

            # Generate the calibration and test dataframes.
            calibration_X, calibration_y, full_calibration_df, test_X, test_y, full_test_df = generate_dataframes(participant_df=participant_df, date_time_column=date_time_column, calibration_start_time=calibration_start_time, calibration_end_time=calibration_end_time,
                                                                                                                  test_start_time=test_start_time, test_end_time=test_end_time, columns_to_exclude=columns_to_exclude, features_to_encode=features_to_encode, prefix_to_include=prefix_to_include,
                                                                                                                  targets=targets, interaction_terms=interaction_terms, scale_explanatory_variables=scale_explanatory_variables)

            if calibration_X is None or calibration_y is None or full_calibration_df is None or test_X is None or test_y is None or full_test_df is None:
                calibration_len = len(full_calibration_df) if full_calibration_df is not None else 0
                test_len = len(full_test_df) if full_test_df is not None else 0
                message = "forecast day, " + str(start_time) + ", " + str(
                    calibration_len) + " calibration observations, " + str(test_len) + " test observations. skipping."
                print(message)
                info_messages.append(message)
                start_time = start_time + td(days=1)
                continue

            if len(full_calibration_df) < min_number_of_calibration_points:
                message = "Only have " + str(len(full_calibration_df)) + " datapoints to calibrate against for calibration_start_time: " + str(
                    calibration_start_time) + ", calibration_end_time: " + str(calibration_end_time) + ", will try naive forecast."
                print(message)
                info_messages.append(message)
                output, train_mae_dictionary = multitarget_insufficient_calibration_data(model_str=model_str, Y_train=calibration_y, tr_df=full_calibration_df, Y_test=test_y, t_df=full_test_df, targets=targets, plot=False,
                                                                                         return_test_set_actuals_and_forecast=True, naive_estimator=True, flagged=True, reason="min_observations")
                if len(output) == 0 or not bool(train_mae_dictionary):
                    message = "Error, no forecast produced, test_start_time: " + \
                        str(test_start_time) + ", test_end_time: " + \
                        str(test_end_time) + ", skipping."
                    print(message)
                    error_messages.append(message)
                    start_time = start_time + td(days=1)
                    continue
                test_forecast_and_actuals = pd.concat([test_forecast_and_actuals, output], ignore_index=True)
                train_mae_results["Flagged"].append(train_mae_dictionary["Flagged"])
                train_mae_results["Reason"].append(train_mae_dictionary["Reason"])
                for target in targets:
                    train_mae_results[target].append(train_mae_dictionary[target])
                start_time = start_time + td(days=1)
                continue

            if ffnn:
                m_epochs = global_m_epochs
                nrows, input_cols = calibration_X.shape
                nrows, number_of_targets = calibration_y.shape
                error_type = "mean_absolute_error"
                if iteration == 0 or iteration % 7 == 0:
                    model = create_layered_ffnn_model(number_of_nodes_list=number_of_nodes_list, input_cols=input_cols,
                                                      number_of_targets=number_of_targets, error_type=error_type, drop_out=drop_out, learning_rate=learning_rate)
                iteration += 1

            # Fit the model to the calibration datasest & print metrics.
            # Taking the explanatory variables from the test set, produce a forecast, compare forecast to actuals, print metrics.
            return_test_set_actuals_and_forecast = True
            naive_estimator = naive_estimator_2day_lookback or naive_estimator_7day_lookback
            try:
                output, train_mae_dictionary = fit_multitarget_model_and_plot(model=model, model_str=model_str, X_train=calibration_X, Y_train=calibration_y, tr_df=full_calibration_df,
                                                                              X_test=test_X, Y_test=test_y, t_df=full_test_df, targets=targets, m_epochs=m_epochs, m_verbose=m_verbose, need_to_concat=need_to_concat,
                                                                              plot=False, return_test_set_actuals_and_forecast=return_test_set_actuals_and_forecast, apply_target_scaling=apply_target_scaling,
                                                                              naive_estimator=naive_estimator, naive_estimator_columns=naive_estimator_columns, flagged=False, reason=None,earlystopping=earlystopping)

                if len(output) == 0 or not bool(train_mae_dictionary):
                    message = "Error, no forecast produced, test_start_time: " + \
                        str(test_start_time) + ", test_end_time: " + \
                        str(test_end_time) + ", skipping."
                    print(message)
                    error_messages.append(message)
                    start_time = start_time + td(days=1)
                    iteration += 1
                    continue

                test_forecast_and_actuals = pd.concat([test_forecast_and_actuals, output], ignore_index=True)
                train_mae_results["Flagged"].append(train_mae_dictionary["Flagged"])
                train_mae_results["Reason"].append(train_mae_dictionary["Reason"])
                for target in targets:
                    train_mae_results[target].append(train_mae_dictionary[target])

                start_time = start_time + td(days=1)

                iteration += 1

                if iteration % 7 == 0 and save_test_to_csv and len(test_forecast_and_actuals) > 0:
                    csv_file_path = os.path.join(csv_output_location, model_str + ".csv")
                    test_forecast_and_actuals.to_csv(csv_file_path, index=False)

            except Exception as exc:
                message = "Error, no forecast produced, test_start_time: " + \
                    str(test_start_time) + ", test_end_time: " + \
                    str(test_end_time) + ", message:" + str(exc)
                print(message)
                error_messages.append(message)
                start_time = start_time + td(days=1)
                iteration += 1

        if save_test_to_csv and len(test_forecast_and_actuals) > 0:
            csv_file_path = os.path.join(csv_output_location, model_str + ".csv")
            test_forecast_and_actuals.to_csv(csv_file_path, index=False)

        print("helper_functions, multitarget_walk_forward_validation, end.")

        return train_mae_results, test_forecast_and_actuals, error_messages, info_messages

    except Exception:
        print("Error: helper_functions, multitarget_walk_forward_validation, end")
        traceback.print_exc()
        return dict(), pd.DataFrame(), error_messages, info_messages


def generate_dataframes(participant_df: pd.DataFrame, date_time_column: str, calibration_start_time: dt, calibration_end_time: dt,
                        test_start_time: dt, test_end_time: dt, columns_to_exclude: [str], features_to_encode: [str], prefix_to_include: [str],
                        targets: [str], interaction_terms: [str], scale_explanatory_variables=True):
    """

    This method takes the participant dataframe and converts it into a calibration/training dataframe and a test dataframe.

    Parameters
    ----------
    participant_df : pd.DataFrame
        Pandas dataframe, contains the participant time series info.
    date_time_column : str
        This is the column in the participant_df which indicates the deliveryperiod.
    calibration_start_time : dt
        The calibration dataframe will cover the interval [calibration_start_time, calibration_end_time].
    calibration_end_time : dt
        See previous comment.
    test_start_time : dt
        The testd dataframe will cover the 24 trading periods from [calibration_end_time,calibration_end_time + 24 hours].
    test_end_time : dt
        See previous comment.
    columns_to_exclude: [str]
        These are the columns which we don't want to appear as explanatory variables.
    features_to_encode: [str]
        These are the categorical columns for which we want to apply one hot encoding.
    prefix_to_include: [str]
        For the categorical to which we apply one hot encoding, this list helps inform the naming convention for the newly created columns.
    targets: [str]
        These are the columns that we are trying to predict.

    Returns
    -------
    A tuple of dataframes. The dataframes relate to calibration/training and test time frames.
    calibration_X: this is a dataframe of the explanatory variables, for the calibration period, that have been scaled.
    calibration_y: this is a dataframe of the target variables, for the calibration period.
    full_calibration_df: this is the calibration period dataframe (it contains the previous two dataframes + additional info that will be used for plotting only)
    test_X, test_y, full_test_df: same as the previous 3 lines except that it applies to the test time frame.

    Note: when fitting a model, it will be fit on calibration_X and calibration_y.
          To test the fitted model, it is applied to test_X and the output is compared to test_y.
    """
    calibration_X = None
    calibration_y = None
    full_calibration_df = None
    test_X = None
    test_y = None
    full_test_df = None

    try:

        if len(participant_df) == 0:
            print(
                "Warning: helper_functions, generate_dataframes method, participant_df has 0 rows. Ending.")
            return calibration_X, calibration_y, full_calibration_df, test_X, test_y, full_test_df

        # The original dataframe might not have interaction terms, here we create them.
        if len(interaction_terms) > 0:
            for val in interaction_terms:
                var1 = None
                var2 = None
                if "/" in val:
                    var1, var2 = val.split("/")
                    participant_df[val] = participant_df[var1] / \
                        participant_df[var2]
                else:
                    var1, var2 = val.split(" ")
                    participant_df[val] = participant_df[var1] * \
                        participant_df[var2]

        original_columns = list(participant_df.columns)
        numeric_columns = [
            item for item in original_columns if item not in columns_to_exclude]

        # Apply One hot encoding if necessary
        for val in features_to_encode:
            participant_df[val] = participant_df[val].astype(np.int64)
        for i in range(0, len(features_to_encode)):
            participant_df = encode_and_bind(
                participant_df, features_to_encode[i], prefix_to_include[i])

        # Remove any rows with nan's etc (there shouldn't be any in the input).
        participant_df = participant_df.dropna()

        # Create the calibration dataframe.
        full_calibration_df = None
        calibration_start_time_str = calibration_start_time.strftime(
            "%Y-%m-%d %H:%M:%S")
        calibration_end_time_str = calibration_end_time.strftime(
            "%Y-%m-%d %H:%M:%S")
        full_calibration_df = participant_df[(participant_df[date_time_column] >= calibration_start_time_str) & (
            participant_df[date_time_column] < calibration_end_time_str)]

        if len(full_calibration_df) == 0:
            print("Don't have a calibration dataframe for calibration_start_time: " +
                  calibration_start_time_str + ", calibration_end_time: " + calibration_end_time_str + ", exiting.")
            return calibration_X, calibration_y, full_calibration_df, test_X, test_y, full_test_df

        # Create the test dataframe
        full_test_df = None
        test_start_time_str = test_start_time.strftime("%Y-%m-%d %H:%M:%S")
        test_end_time_str = test_end_time.strftime("%Y-%m-%d %H:%M:%S")
        full_test_df = participant_df[(participant_df[date_time_column] >= test_start_time_str) & (
            participant_df[date_time_column] < test_end_time_str)]

        if len(full_test_df) == 0:
            print("Don't have a test dataframe for test_start_time: " +
                  test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")
            return calibration_X, calibration_y, full_calibration_df, test_X, test_y, full_test_df

        # Next we want to create the dataframes that will feed direclty into the model.
        regression_columns = [item for item in list(
            full_calibration_df.columns) if item not in columns_to_exclude]
        categorical_columns = [
            item for item in regression_columns if item not in numeric_columns]

        calibration_y = full_calibration_df[targets].copy()
        c_X = full_calibration_df[regression_columns].copy()

        test_y = full_test_df[targets].copy()
        t_X = full_test_df[regression_columns].copy()

        # Applying scaling to the numeric explanatory columns.
        if scale_explanatory_variables:
            # scale the explanatory variables in the training set
            scaler = StandardScaler()
            scaled_numeric_calibration_df = pd.DataFrame(scaler.fit_transform(
                c_X[numeric_columns]), columns=numeric_columns, index=c_X.index)
            temp_df_calibration = c_X[categorical_columns]
            calibration_X = pd.concat(
                [temp_df_calibration, scaled_numeric_calibration_df], axis=1)
            # scale the explanatory variables in the test set
            scaled_numeric_test_df = pd.DataFrame(scaler.transform(
                t_X[numeric_columns]), columns=numeric_columns, index=t_X.index)
            temp_df_test = t_X[categorical_columns]
            test_X = pd.concat([temp_df_test, scaled_numeric_test_df], axis=1)
        else:
            calibration_X = c_X.copy()
            test_X = t_X.copy()

        return calibration_X, calibration_y, full_calibration_df, test_X, test_y, full_test_df

    except Exception:
        print("Error: helper_functions, generate_dataframes method.")
        traceback.print_exc()
        return calibration_X, calibration_y, full_calibration_df, test_X, test_y, full_test_df


def create_layered_ffnn_model(number_of_nodes_list: [int], input_cols: int, number_of_targets: int, error_type: str, drop_out=0.2, learning_rate=0.001):
    """
    Creates the basic setup of the tensorflow/keras feed forward neural network model.

    Parameters
    ----------
    number_of_nodes_list : [int]
        Vector of ints, each int describes the number of nodes present in a layer.
    input_cols : int
        The number of columns in the input dataframe.
    number_of_targets : int
        The number of targets in the multitarget vector i.e. the number of outputs in each prediction/forecast.
    error_type : str
        Whether we are using the mae or mse.

    Returns
    -------
    Sequential model.

    """
    try:

        print("helper_functions, create_layered_ffnn_model method, start.")

        model = models.Sequential()

        initializer = he_uniform(seed=42)

        if len(number_of_nodes_list) == 1:
            #        model.add(Dropout(drop_out,input_shape=(input_cols,)))
            model.add(layers.Dense(number_of_nodes_list[0], activation='relu', input_shape=(
                input_cols,), use_bias=True, kernel_initializer=initializer, kernel_regularizer=keras.regularizers.l2(0.01)))
            model.add(layers.BatchNormalization())

            model.add(layers.Dense(number_of_targets,
                                   kernel_regularizer=keras.regularizers.l2(0.01)))

        if len(number_of_nodes_list) > 1:

            #            model.add(Dropout(drop_out,input_shape=(input_cols,)))
            model.add(layers.Dense(number_of_nodes_list[0], activation='relu', input_shape=(
                input_cols,), use_bias=True, kernel_initializer=initializer, kernel_regularizer=keras.regularizers.l2(0.01)))
            model.add(layers.BatchNormalization())

            for i in range(1, len(number_of_nodes_list)):
                #                model.add(Dropout(drop_out))
                model.add(layers.Dense(number_of_nodes_list[i], activation='relu', use_bias=True,
                                       kernel_initializer=initializer, kernel_regularizer=keras.regularizers.l2(0.01)))
                model.add(layers.BatchNormalization())

            model.add(layers.Dense(number_of_targets,kernel_regularizer=keras.regularizers.l2(0.01)))

        if error_type == "mean_squared_error":
            print("helper_functions, create_layered_ffnn_model, using mean_squared_error metric.")
            optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])
        else:
            print("helper_functions, create_layered_ffnn_model, using mean_absolute_error metric.")
            optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
            model.compile(optimizer=optimizer, loss='mae',metrics=['mean_absolute_error'])

        print("helper_functions, create_layered_ffnn_model method, end.")

        return model

    except Exception:
        print("Error: helper_functions, create_layered_ffnn_model method.")
        traceback.print_exc()


def fit_multitarget_model_and_plot(model, model_str, X_train, Y_train, tr_df, X_test, Y_test, t_df, targets,
                                   m_epochs=None, batch_size=None, m_verbose=None, need_to_concat=False, plot=True, return_test_set_actuals_and_forecast=True,
                                   apply_target_scaling=False, naive_estimator=None, naive_estimator_columns=None, flagged=False, reason=None,earlystopping=None):
    """
    Fits the model to the calibration data, prints model statistics and produces plots.
    Once the model has been fit to the calibration data, it applies it to the test set, prints test set metrics and outputs associated plots.

    Parameters
    ----------
    model : object
        Model object i.e. randomforestregressor, ffnn or other.
    model_str : str
        String used in the plot titles.
    X_train : dataframe
        The explanatory variables for the train/calibration set, any numeric columns may already have been scaled.
    Y_train : dataframe
        The target variables for the train/calibration set.
    tr_df : dataframe
        This is the full train/calibration dataset. It is equal to X_train_scaled + Y_train + other info. This dataset will be used when plotting.
    X_test : dataframe
        The explanatory variables for the test set, any numeric columns may be scaled. It will comprise of 24 rows (1 row for each delivery period in the trading day).
    Y_test : dataframe
        The target variables for the test set. Similar to the previous bullet point, the dataframe will contain 24 rows.
    t_df : dataframe
        Similar comment as to the tr_df except for the test set.
    predictor_str : [str]
        These are the target vectors that we want to predict/forecast.
    m_epochs: int
       Nunumber of epochs. This variable will be used if we are fitting a neural network.
    batch_size: int
       This variable will be used if we are fitting a neural network. Batch size if the model is a neural network.
    need_to_concat: boolean
       Applicable when fitting a NN type model. The predict output is in a different format to the sklearn predict output
    plot: boolean
       As in fit_model_and_plot_function
    return_test_set_actuals_and_forecast: boolean
       If true, returns the actual and forecast values for the test set.
    plot_3d: boolean
       Whether or not to do the 3d plot
    apply_target_scaling: boolean
       The target vector, do we want to scale it?
    naive_estimator: boolean.
       Whether or not to use a naive estimator.
    naive_estimator_columns: [str].
       If a anive estimator is used, this is the list of columns which will produce the forecast.
    flagged: boolean.
       Indicator variable used in the output file. For example, one use case is where there mightn't have been enough observations to calibrate a model and hence the forecast is flagged.
    reason: str.
       If flagged is true, the reason for the flagging.

    Returns
    -------
    If return_test_set_actuals_and_forecast is true, a dataframe is returned. The dataframe will contain the delivery period and actual/forecast values
    for the test set.

    """

    try:
        # set_trace()
        scaler = StandardScaler()

        # neural network type models i.e. tf and keras
        if m_epochs is not None and m_verbose is not None and model is not None and not naive_estimator:

            if apply_target_scaling:
                Y_train_df = pd.DataFrame(data=scaler.fit_transform(Y_train), index=Y_train.index, columns=Y_train.columns)
                if earlystopping is not None:
                    rows_to_train = len(X_train) - 7*24
                    X_last = X_train.tail(7*24)
                    X_first = X_train.head(rows_to_train)
                    Y_last = Y_train_df.tail(7*24)
                    Y_first = Y_train_df.head(rows_to_train)
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10,restore_best_weights=True)
                    model.fit(X_first, Y_first, validation_data=(X_last, Y_last), epochs=m_epochs, batch_size=batch_size, verbose=m_verbose, callbacks=[es])
                else:
                    model.fit(X_train, Y_train_df, epochs=m_epochs, batch_size=batch_size, verbose=m_verbose)

            else:
                model.fit(X_train, Y_train, epochs=m_epochs,batch_size=batch_size, verbose=m_verbose)

        # sklearn models i.e. RF, SVR etc.
        elif model is not None and not naive_estimator:
            if apply_target_scaling:
                Y_train_df = pd.DataFrame(data=scaler.fit_transform(
                    Y_train), index=Y_train.index, columns=Y_train.columns)
                model.fit(X_train, Y_train_df)
            else:
                model.fit(X_train, Y_train)

        model_train_predictions = None
        model_test_predictions = None

        train_mae_dictionary = dict()

        if apply_target_scaling and model is not None and not naive_estimator:
            model_train_predictions = scaler.inverse_transform(model.predict(X_train))
            model_test_predictions = scaler.inverse_transform(model.predict(X_test))
        elif model is not None and not naive_estimator:
            model_train_predictions = model.predict(X_train)
            model_test_predictions = model.predict(X_test)
        # Naive Model is just the target variable at a particular lag.
        elif naive_estimator:
            model_train_predictions = tr_df[naive_estimator_columns].to_numpy()
            model_test_predictions = t_df[naive_estimator_columns].to_numpy()

        print("train number of observations: " + str(len(Y_train)))
        model_train_mse = mean_squared_error(Y_train, model_train_predictions)
        model_train_rmse = round(np.sqrt(model_train_mse), 2)
        model_train_mae = round(mean_absolute_error(
            Y_train, model_train_predictions), 2)
        print(model_str + " train rmse: " + str(model_train_rmse))
        print(model_str + " train mae: " + str(model_train_mae))

        cols = Y_train.columns.values.tolist()
        train_mae_dictionary["Flagged"] = flagged
        train_mae_dictionary["Reason"] = reason
        for i in range(0, len(cols)):
            tr_df[cols[i]+"_Forecast"] = model_train_predictions[:, i].tolist()
            predictor_train_mse = mean_squared_error(
                Y_train[cols[i]], model_train_predictions[:, i])
            predictor_train_rmse = round(np.sqrt(predictor_train_mse), 2)
            predictor_train_mae = round(mean_absolute_error(
                Y_train[cols[i]], model_train_predictions[:, i]), 2)
            train_mae_dictionary[cols[i]] = predictor_train_mae
            print(model_str + " " +
                  cols[i] + " train rmse: " + str(predictor_train_rmse))
            print(model_str + " " + cols[i] +
                  " train mae: " + str(predictor_train_mae))

        # plot the calibration/train data.
        if plot:
            tr_df.sort_index(inplace=True)
            for i in range(0, len(cols)):
                title_str = model_str + " " + cols[i] + " Calibration set"
                tr_df.plot(y=[cols[i]+"_Forecast", cols[i]],
                           x="DeliveryPeriod", style=['bs-', 'ro-'], title=title_str)

        print("test number of observations: " + str(len(Y_test)))
        model_test_mse = mean_squared_error(Y_test, model_test_predictions)
        model_test_rmse = round(np.sqrt(model_test_mse), 2)
        model_test_mae = round(mean_absolute_error(
            Y_test, model_test_predictions), 2)
        print(model_str + " test rmse: " + str(model_test_rmse))
        print(model_str + " test mae: " + str(model_test_mae))

        flagged_vector = [flagged] * len(t_df)
        reason_vector = [reason] * len(t_df)
        t_df["Flagged"] = flagged_vector
        t_df["Reason"] = reason_vector
        for i in range(0, len(cols)):
            t_df[cols[i]+"_Forecast"] = model_test_predictions[:, i].tolist()
            predictor_test_mse = mean_squared_error(
                Y_test[cols[i]], model_test_predictions[:, i])
            predictor_test_rmse = round(np.sqrt(predictor_test_mse), 2)
            predictor_test_mae = round(mean_absolute_error(
                Y_test[cols[i]], model_test_predictions[:, i]), 2)
            print(model_str + " " + cols[i] +
                  " test rmse: " + str(predictor_test_rmse))
            print(model_str + " " + cols[i] +
                  " test mae: " + str(predictor_test_mae))

        # plot the test set
        if plot:
            t_df.sort_index(inplace=True)
            for i in range(0, len(cols)):
                title_str = model_str + " " + cols[i] + " Test Set"
                t_df.plot(y=[cols[i]+"_Forecast", cols[i]],
                          x="DeliveryPeriod", style=['bs-', 'ro-'], title=title_str)

        if return_test_set_actuals_and_forecast:
            t_df.sort_index(inplace=True)
            test_columns = ["DeliveryPeriod", "Flagged", "Reason"]
            for i in range(0, len(cols)):
                test_columns.extend([cols[i]+"_Forecast", cols[i], cols[i]+"_-48"])
            t_df = t_df[test_columns]

        return t_df, train_mae_dictionary

    except Exception:
        print("Error: helper_functions, fit_multitarget_model_and_plot method.")
        traceback.print_exc()
        return pd.DataFrame(), dict()


def multitarget_insufficient_calibration_data(model_str, Y_train, tr_df, Y_test, t_df, targets, plot=True,
                                              return_test_set_actuals_and_forecast=True, naive_estimator=True, flagged=True, reason="min_observations"):
    """
    This method is used when there are < min_number_of_calibration_points in the calibration dataframe.
    In that scenario, we utilise the 2 day lookback values for the forecast (the info will be in the dataframe as an explanatory variable).
    If the 2 day lookback values aren't available, instead take the corresponding values from 7 days earlier.

    Parameters
    ----------
    model_str : str
        String used in the plot titles.
    Y_train : dataframe
        The target variables for the train/calibration set.
    tr_df : dataframe
        This is the full train/calibration dataset. It is equal to X_train_scaled + Y_train + other info. This dataset will be used when plotting.
    Y_test : dataframe
        The target variables for the test set. Similar to the previous bullet point, the dataframe will contain 24 rows.
    t_df : dataframe
        Similar comment as to the tr_df except for the test set.
    predictor_str : [str]
        These are the target vectors that we want to predict/forecast.
    plot: boolean
       As in fit_model_and_plot_function
    return_test_set_actuals_and_forecast: boolean
       If true, returns the actual and forecast values for the test set.
    flagged: boolean.
       Indicator variable used in the output file. For example, one use case is where there mightn't have been enough observations to calibrate a model and hence the forecast is flagged.
    reason: str.
       If flagged is true, the reason for the flagging.

    Returns
    -------
    If return_test_set_actuals_and_forecast is true, a dataframe is returned. The dataframe will contain the delivery period and actual/forecast values
    for the test set.
    """
    try:

        hours = [-48, -168]

        for h in hours:

            model_train_predictions = None
            model_test_predictions = None

            train_mae_dictionary = dict()

            tr_df_copy = tr_df.copy()
            t_df_copy = t_df.copy()

            naive_estimator_columns = [x+"_"+str(h) for x in targets]

            model_train_predictions = tr_df_copy[naive_estimator_columns].to_numpy()
            model_test_predictions = t_df_copy[naive_estimator_columns].to_numpy()

            print("train number of observations: " + str(len(Y_train)))
            model_train_mse = mean_squared_error(
                Y_train, model_train_predictions)
            model_train_rmse = round(np.sqrt(model_train_mse), 2)
            model_train_mae = round(mean_absolute_error(
                Y_train, model_train_predictions), 2)
            print(model_str + " train rmse: " + str(model_train_rmse))
            print(model_str + " train mae: " + str(model_train_mae))

            cols = Y_train.columns.values.tolist()
            train_mae_dictionary["Flagged"] = flagged
            train_mae_dictionary["Reason"] = reason
            for i in range(0, len(cols)):
                tr_df_copy[cols[i] +
                           "_Forecast"] = model_train_predictions[:, i].tolist()
                predictor_train_mse = mean_squared_error(
                    Y_train[cols[i]], model_train_predictions[:, i])
                predictor_train_rmse = round(np.sqrt(predictor_train_mse), 2)
                predictor_train_mae = round(mean_absolute_error(
                    Y_train[cols[i]], model_train_predictions[:, i]), 2)
                train_mae_dictionary[cols[i]] = predictor_train_mae
                print(model_str + " " +
                      cols[i] + " train rmse: " + str(predictor_train_rmse))
                print(model_str + " " +
                      cols[i] + " train mae: " + str(predictor_train_mae))

            print("test number of observations: " + str(len(Y_test)))
            model_test_mse = mean_squared_error(Y_test, model_test_predictions)
            model_test_rmse = round(np.sqrt(model_test_mse), 2)
            model_test_mae = round(mean_absolute_error(
                Y_test, model_test_predictions), 2)
            print(model_str + " test rmse: " + str(model_test_rmse))
            print(model_str + " test mae: " + str(model_test_mae))

            flagged_vector = [flagged] * len(t_df_copy)
            reason_vector = [reason] * len(t_df_copy)
            t_df_copy["Flagged"] = flagged_vector
            t_df_copy["Reason"] = reason_vector
            for i in range(0, len(cols)):
                t_df_copy[cols[i] +
                          "_Forecast"] = model_test_predictions[:, i].tolist()
                predictor_test_mse = mean_squared_error(
                    Y_test[cols[i]], model_test_predictions[:, i])
                predictor_test_rmse = round(np.sqrt(predictor_test_mse), 2)
                predictor_test_mae = round(mean_absolute_error(
                    Y_test[cols[i]], model_test_predictions[:, i]), 2)
                print(model_str + " " +
                      cols[i] + " test rmse: " + str(predictor_test_rmse))
                print(model_str + " " +
                      cols[i] + " test mae: " + str(predictor_test_mae))

            # plot the test set
            if plot:
                t_df_copy.sort_index(inplace=True)
                for i in range(0, len(cols)):
                    title_str = model_str + " " + cols[i] + " Test Set"
                    t_df_copy.plot(y=[cols[i]+"_Forecast", cols[i]],
                                   x="DeliveryPeriod", style=['bs-', 'ro-'], title=title_str)

            if return_test_set_actuals_and_forecast:
                t_df_copy.sort_index(inplace=True)
                test_columns = ["DeliveryPeriod", "Flagged", "Reason"]
                for i in range(0, len(cols)):
                    test_columns.extend(
                        [cols[i]+"_Forecast", cols[i], cols[i]+"_-48"])
                t_df_copy = t_df_copy[test_columns]

            if len(t_df_copy) > 0 and bool(train_mae_dictionary):
                return t_df_copy, train_mae_dictionary

        return pd.DataFrame(), dict()

    except Exception:
        print("Error: helper_functions, multitarget_insufficient_calibration_data method.")
        traceback.print_exc()
        return pd.DataFrame(), dict()


def encode_and_bind(original_dataframe: str, feature_to_encode: str, prefix_to_include: str):
    dummies = pd.get_dummies(
        original_dataframe[feature_to_encode], prefix=prefix_to_include)
    res = pd.concat([original_dataframe, dummies], axis=1)
    return res



