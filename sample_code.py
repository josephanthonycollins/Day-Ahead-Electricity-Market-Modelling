# -*- coding: utf-8 -*-
"""
Created on 28th December 2024.

@author: Joseph Collins, Part-time Phd Student, University College Cork, Department of Applied Mathematics, email 98718584@umail.ucc.ie, Supervisors = {Dr. Andreas Amann, Dr. Kieran Mulchrone}

Purpose and Context: The code in this file is sample/mock code associated with the research paper "Using Granular Participant-Level Order Data to Forecast a Day-Ahead Electricity Market".
                     The sole purpose of this file is to illustrate the forecasting process.
                     If it is to be used in practice, then additional changes are required (for example some of the functions assume access to a private mysql database which 
                     is used to keep track of model runs, this and other aspects would need to be changed.)
"""

import datetime as dt
from datetime import datetime
import csv
from helper_functions import multitarget_walk_forward_validation_wrapper

if __name__ == "__main__":
                

#################################################################################Common#########################################################################################################################################################################################

    # List of participants.
    unit_names = ['GU_400271', 'GU_403930', 'AU_400148', 'GU_402220', 'GU_501710', 'AU_500122', 'AU_400139', 'SU_400339', 'SU_500076', 'GU_400021', 'DSU_401800', 'DSU_403450', 'GU_401690', 'GU_400671', 'GU_403900', 'SU_400201', 'GU_401090', 'GU_400250', 'SU_400196', 'GU_400362', 'GU_501700', 'AU_400137', 'AU_400004', 'DSU_403040', 'SU_400233', 'SU_500075', 'AU_400147', 'SU_400040', 'GU_501440', 'GU_404330', 'GU_401720', 'SU_400137', 'GU_503380', 'SU_400386', 'GU_405130', 'SU_500020', 'SU_400188', 'AU_400106', 'GU_402230', 'GU_402010', 'SU_400366', 'SU_400134', 
              'GU_401820', 'GU_503810', 'AU_400130', 'GU_503960', 'GU_501500', 'GU_403770', 'GU_503790', 'GU_400180', 'AU_400017', 'GU_400183', 'GU_400630', 'GU_400750', 'AU_500003', 'GU_403990', 'SU_400100', 'DSU_401400', 'SU_400358', 'SU_403030', 'AU_500111', 'SU_400216', 'AU_400144', 'SU_403020', 'GU_503930', 'AU_400135', 'GU_501130', 'GU_400751', 'GU_401960', 'AU_400123', 'AU_400119', 'GU_400281', 'SU_400071', 'SU_403010', 'GU_404420', 'GU_400660', 'GU_400600', 'GU_403920', 'DSU_403470', 'AU_400020', 'SU_400161', 'DSU_401880', 'SU_400336', 'GU_501660', 'GU_400252', 'GU_405140', 'GU_400030', 'GU_503740', 'AU_500105', 'GU_400050',
              'GU_400272', 'SU_400041', 'GU_400731', 'SU_500060', 'SU_500082', 'GU_500220', 'GU_501050', 'AU_400010', 'GU_401750', 'SU_400044', 'GU_403560', 'GU_500821', 'AU_400142', 'SU_400208', 'GU_503200', 'GU_401740', 'GU_405110', 'DSU_403560', 'GU_501650', 'SU_400206', 'AU_500127', 'GU_400950', 'GU_401780', 'GU_400181', 'GU_400753', 'GU_401830', 'GU_405080', 'DSU_401850', 'GU_504070', 'GU_501620', 'SU_400042', 'GU_400850', 'GU_405870', 'GU_400650', 'AU_400009', 'GU_400363', 'AU_400113', 'GU_405830', 'GU_503800', 'GU_400591', 'GU_400930', 'GU_500825', 'DSU_403120', 'GU_405930', 'GU_402250', 'GU_401380', 'AU_400111', 'GU_400210', 'GU_503830', 'GU_403880', 'GU_403780', 'AU_400132', 'DSU_402100', 'GU_403830', 'SU_400189', 'AU_500001', 'SU_400155',
              'DSU_501450', 'GU_503720', 'GU_401840', 'AU_400118', 'GU_401890', 'GU_500130', 'AU_500126', 'GU_400211', 'GU_501410', 'GU_400940', 'GU_402270', 'GU_400530', 'DSU_401270', 'GU_400130', 
              'SU_500085', 'GU_501210', 'GU_402110', 'GU_400070', 'AU_400110', 'SU_403040', 'DSU_403520', 'AU_500114', 'GU_400041', 'GU_400480', 'SU_400195', 'GU_405150', 'GU_500020', 'GU_400360', 'SU_400070', 'GU_400430', 'GU_401010', 'GU_503820', 'AU_400105', 'GU_400370', 'GU_501480', 'GU_400781', 'GU_402260', 'SU_400191', 'GU_501750', 'AU_500121', 'SU_500311', 'GU_401900', 'SU_400130', 'GU_404000', 'AU_500104', 'SU_400190', 'SU_400135', 'SU_500051', 'GU_402320', 'GU_503370', 'GU_401980', 'SU_400224', 'AU_400150', 'GU_404460', 'DSU_402120', 'AU_400101', 'GU_500211', 'GU_403810', 'AU_500008', 'SU_400046', 'SU_400346', 'GU_404520', 'GU_402350', 'GU_500720', 'AU_400002', 'GU_400610', 'GU_401670', 'SU_400176', 'AU_400112', 'GU_401710', 'AU_400129', 'GU_500284', 'SU_400151', 'AU_500128', 'AU_400103', 'GU_401640', 'AU_400138', 'GU_503910', 'GU_501720', 'AU_400015', 'SU_400228', 'DSU_401910', 'GU_402210', 'GU_401020', 'GU_400020', 'GU_400290', 'GU_503730', 'DSU_403080', 'AU_400128', 'GU_503850', 'GU_401630', 'SU_400207', 'SU_400083', 'GU_404560', 'GU_405190', 'GU_500283', 'AU_500125', 'SU_400043', 'GU_405210', 'SU_400049', 'GU_403000', 'AU_400116', 'DSU_402090', 'GU_400780', 'GU_501160', 'GU_500281', 'SU_400020', 'GU_402170', 'GU_403790', 'GU_401700', 'AU_400151', 'GU_501400', 'GU_400080', 'GU_501490', 'GU_501670', 'GU_401930', 'GU_501690', 'GU_501520', 'GU_402190', 'AU_400003', 'GU_503900', 'AU_400005', 'GU_401460', 'SU_400183', 'SU_400355', 'AU_500012', 'GU_500740', 'GU_400280', 'GU_503400', 'GU_404210', 'GU_504050', 'SU_400349', 'GU_403390', 'GU_400540', 'GU_400202', 'GU_404510', 'GU_400620', 'GU_404990', 'SU_400314', 'AU_400134', 'GU_400325', 'GU_400201', 'SU_403000', 'GU_503310', 'GU_404470', 'DSU_403630', 'GU_403980', 'SU_400335', 'GU_405880', 'GU_401770', 'GU_401440', 'DSU_503460', 'DSU_403050', 'GU_401810', 'AU_500134', 'GU_501430', 'GU_400840', 'GU_403380', 'GU_503950', 'GU_501150', 'AU_400141', 'GU_401950', 'SU_400223', 'AU_400008', 'GU_401970', 'SU_500040', 'GU_404450', 'GU_400220', 'GU_400221', 'GU_503780', 'AU_400122', 'GU_402020', 'GU_404270', 'SU_400214', 'SU_400360', 'SU_400045', 'AU_400140', 'AU_400143', 'AU_400011', 'SU_400192', 'GU_400200', 'GU_400920', 'GU_403820', 'GU_401860', 'GU_402300', 'AU_400006', 'GU_500822', 'GU_401080', 
              'SU_500077', 'SU_400048', 'GU_503210', 'GU_402240', 'GU_400810', 'TU_400001', 'GU_402080', 'DSU_403030', 'AU_400126', 'SU_400132', 'GU_400970', 'GU_501040', 'AU_500102', 'SU_400352', 'AU_400131', 'GU_402150', 'SU_403050', 'GU_401500', 'GU_400670', 'GU_501360', 'GU_501230', 'GU_404340', 'GU_403950', 'GU_401920', 'GU_504060', 'GU_401260', 'AU_500009', 'GU_400500', 'SU_400219', 'SU_400177', 'AU_400117', 'GU_503940', 'SU_500309', 'GU_501680', 'GU_400060', 'GU_403800', 'GU_402340', 'GU_401170', 'GU_401680', 'GU_400203', 'GU_403940', 'GU_404480', 'GU_504040', 'GU_500131', 'SU_400047', 'DSU_401620', 'SU_400347', 'GU_501740', 'GU_403910', 'GU_404260', 'GU_501630', 'GU_400910', 'SU_400147', 'SU_400136', 'SU_400173', 'DSU_403640', 'GU_503840', 'GU_405890', 'GU_401230', 'GU_501190', 'GU_501120', 'SU_400138', 'AU_400001', 'GU_400324', 'AU_400014', 'GU_501550', 'GU_403760', 'GU_400260', 'GU_404500', 'AU_500101', 'GU_404290', 'GU_400550', 'AU_500131', 'GU_400762', 'GU_400470', 
              'GU_501730', 'GU_404550', 'AU_500135', 'SU_400081', 'GU_404580', 'GU_402160', 'GU_401990', 'GU_401730', 'GU_405120', 'GU_403750', 'SU_400133', 'GU_401940', 'AU_500116', 'GU_400730', 'GU_400182', 'GU_405160', 'GU_500820', 'GU_500824', 'GU_500040', 'GU_504200', 'GU_500140', 'GU_501350', 'SU_500308', 'AU_400120', 'GU_402310', 'SU_400205', 'SU_400337', 'GU_400240', 'AU_500118', 'GU_400752', 'GU_500010', 'DSU_401610', 'DSU_403020', 'GU_504160', 'GU_400361', 'GU_401540', 'SU_400343', 'SU_400185', 'AU_400136', 'GU_404490', 'GU_404570', 'AU_400114', 'AU_500115', 'GU_500823', 
              'GU_500904', 'AU_400100', 'DSU_403760', 'AU_500110', 'GU_400270', 'SU_400330', 'GU_504090', 'SU_400354', 'GU_400771', 'GU_402050', 'GU_500041', 'GU_500270', 'SU_400213', 'GU_403840', 'GU_401790', 'GU_403400', 'AU_400125', 'SU_400329', 'GU_400770', 'AU_400133', 'GU_402030', 'SU_400215', 'GU_504030', 'AU_500113', 'AU_500138', 'GU_401760', 'GU_500282', 'SU_500050', 'GU_402130', 'AU_500109', 'SU_500095', 'GU_404300', 'GU_404010', 'GU_402140', 'GU_401011', 'DSU_401870', 'GU_501390', 'GU_401350', 'GU_503390', 'GU_501310', 'GU_400311', 'GU_400251', 'SU_400193', 'SU_400353', 'GU_500210']        

    #The forecasting horizon.
    start_time = dt.datetime(2020, 10, 1, 23, 0, 0, 0)
    end_time = dt.datetime(2023, 5, 31, 23, 0, 0, 0)


    #we don't produce a forecast for trading periods falling within the clock change (i.e. date convention in some of the source files for these days may not align). 
    daylightsavings_day_starttime = [dt.datetime(2019, 3, 30, 23, 0, 0, 0), dt.datetime(2019, 10, 26, 23, 0, 0, 0), dt.datetime(2020, 3, 28, 23, 0, 0, 0), 
                                     dt.datetime(2020, 10, 24, 23, 0, 0, 0), dt.datetime(2021, 3, 27, 23, 0, 0, 0), dt.datetime(2021, 10, 30, 23, 0, 0, 0),
                                     dt.datetime(2022, 3, 26, 23, 0, 0, 0), dt.datetime(2022, 10, 29, 23, 0, 0, 0), dt.datetime(2023, 3, 25, 23, 0, 0, 0),
                                     dt.datetime(2023, 10, 28, 23, 0, 0, 0)]
    daylightsavings_day_endtime = [dt.datetime(2019, 3, 31, 22, 0, 0, 0), dt.datetime(2019, 10, 27, 22, 0, 0, 0), dt.datetime(2020, 3, 29, 22, 0, 0, 0),
                                   dt.datetime(2020, 10, 25, 22, 0, 0, 0), dt.datetime(2021, 3, 28, 22, 0, 0, 0), dt.datetime(2021, 10, 31, 22, 0, 0, 0),
                                   dt.datetime(2022, 3, 27, 22, 0, 0, 0), dt.datetime(2022, 10, 30, 22, 0, 0, 0), dt.datetime(2023, 3, 26, 22, 0, 0, 0),
                                   dt.datetime(2023, 10, 29, 22, 0, 0, 0)]
    clock_change_start = daylightsavings_day_starttime.copy() 
    clock_change_end = daylightsavings_day_endtime.copy() 

    #Common to all.
    min_number_of_calibration_points = 336 # i.e. two weeks worth of data to calibrate a model at a minimum, otherwise use a naive forecast.
    save_summary_to_db = True 
    save_test_to_csv = True
    plot = False
    
#################################################################################Naive Forecast#########################################################################################################################################################################################
        

    #Context: For each participant the Naive forecast is simply the target data but at a lag of 48 hours. 
    #         That is, if we want to predict the target data for DeliveryPeriod, then we use the target data from [DeliveryPeriod - 48 hours] as the estimate.
    #         Alternative Naive approaches are possible (e.g. lag of 168 hours etc)
    
    approach_1_n_equal_4_naive = False
    
    if approach_1_n_equal_4_naive:
        name = "approach_1_n_equal_4_naive"
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             csv_output_location = raw_csv_location + name
             naive_estimator_columns = ['Buy_Price_0_-48','Buy_Volume_0_-48','Buy_Price_1_-48','Buy_Volume_1_-48','Buy_Price_2_-48','Buy_Volume_2_-48','Buy_Price_3_-48','Buy_Volume_3_-48','Sell_Price_0_-48','Sell_Volume_0_-48','Sell_Price_1_-48','Sell_Volume_1_-48','Sell_Price_2_-48','Sell_Volume_2_-48','Sell_Price_3_-48','Sell_Volume_3_-48']
             #The following lines aren't significant when we run a Naive model.
             targets = ['Buy_Price_0','Buy_Volume_0','Buy_Price_1','Buy_Volume_1','Buy_Price_2','Buy_Volume_2','Buy_Price_3','Buy_Volume_3','Sell_Price_0','Sell_Volume_0','Sell_Price_1','Sell_Volume_1','Sell_Price_2','Sell_Volume_2','Sell_Price_3','Sell_Volume_3']
             explanatory_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'Wind-24', 'Wind-48', 'Wind-72', 'Wind-168', 'DemandAggregatedForecast', 'Demand-24', 'Demand-48', 'Demand-72', 'Demand-168', 'MULTI_FUEL_-24', 'GAS_-24', 'COAL_-24', 'MULTI_FUEL_-168', 'GAS_-168', 'COAL_-168', 'InterconnectorNetTotal_-24', 'InterconnectorNetTotal_-168', 'GasPrice', 'GasPrice_-24', 'GasPrice_-168', 'CarbonPrice', 'CarbonPrice_-24', 'CarbonPrice_-168', 'Buy_Price_0_-48', 'Buy_Volume_0_-48', 'Buy_Price_1_-48', 'Buy_Volume_1_-48', 'Buy_Price_2_-48', 'Buy_Volume_2_-48', 'Buy_Price_3_-48', 'Buy_Volume_3_-48', 'Sell_Price_0_-48', 'Sell_Volume_0_-48', 'Sell_Price_1_-48', 'Sell_Volume_1_-48', 'Sell_Price_2_-48', 'Sell_Volume_2_-48', 'Sell_Price_3_-48', 'Sell_Volume_3_-48', 'PeriodExecutedQuantity_-48', 'MeteredMW_-48', 'IDA1PeriodExecutedQuantity_-48', 'IDA2PeriodExecutedQuantity_-48', 'IDA3PeriodExecutedQuantity_-48', 'Buy_Price_0_-168', 'Buy_Volume_0_-168', 'Buy_Price_1_-168', 'Buy_Volume_1_-168', 'Buy_Price_2_-168', 'Buy_Volume_2_-168', 'Buy_Price_3_-168', 'Buy_Volume_3_-168', 'Sell_Price_0_-168', 'Sell_Volume_0_-168', 'Sell_Price_1_-168', 'Sell_Volume_1_-168', 'Sell_Price_2_-168', 'Sell_Volume_2_-168', 'Sell_Price_3_-168', 'Sell_Volume_3_-168', 'PeriodExecutedQuantity_-168', 'MeteredMW_-168', 'IDA1PeriodExecutedQuantity_-168', 'IDA2PeriodExecutedQuantity_-168', 'IDA3PeriodExecutedQuantity_-168']
             interaction_terms = []
             features_to_encode = []
             model_dictionary = {"ffnn" : False, "svm": False, "rf": False, "knn" : False, "naive_estimator_2day_lookback":True,"naive_estimator_7day_lookback":False}             
             meta_settings = [{}]        
             model_settings = [{}]

             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')

#################################################################################RF Forecasts#########################################################################################################################################################################################
                
    #Context: in this section we generate participant forecasts using random forest modelling approaches.      
    
    approach_1_n_equal_4_rf_a = False

    if approach_1_n_equal_4_rf_a:
        name = "approach_1_n_equal_4_rf_a"
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             csv_output_location = raw_csv_location + name
             targets = ['Buy_Price_0','Buy_Volume_0','Buy_Price_1','Buy_Volume_1','Buy_Price_2','Buy_Volume_2','Buy_Price_3','Buy_Volume_3','Sell_Price_0','Sell_Volume_0','Sell_Price_1','Sell_Volume_1','Sell_Price_2','Sell_Volume_2','Sell_Price_3','Sell_Volume_3']
             explanatory_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'Wind-24', 'Wind-48', 'Wind-72', 'Wind-168', 'DemandAggregatedForecast', 'Demand-24', 'Demand-48', 'Demand-72', 'Demand-168', 'MULTI_FUEL_-24', 'GAS_-24', 'COAL_-24', 'MULTI_FUEL_-168', 'GAS_-168', 'COAL_-168', 'InterconnectorNetTotal_-24', 'InterconnectorNetTotal_-168', 'GasPrice', 'GasPrice_-24', 'GasPrice_-168', 'CarbonPrice', 'CarbonPrice_-24', 'CarbonPrice_-168','Buy_Price_0_-48', 'Buy_Volume_0_-48', 'Buy_Price_1_-48', 'Buy_Volume_1_-48', 'Buy_Price_2_-48', 'Buy_Volume_2_-48', 'Buy_Price_3_-48', 'Buy_Volume_3_-48', 'Sell_Price_0_-48', 'Sell_Volume_0_-48', 'Sell_Price_1_-48', 'Sell_Volume_1_-48', 'Sell_Price_2_-48', 'Sell_Volume_2_-48', 'Sell_Price_3_-48', 'Sell_Volume_3_-48', 'PeriodExecutedQuantity_-48', 'MeteredMW_-48', 'IDA1PeriodExecutedQuantity_-48', 'IDA2PeriodExecutedQuantity_-48', 'IDA3PeriodExecutedQuantity_-48', 'Buy_Price_0_-168', 'Buy_Volume_0_-168', 'Buy_Price_1_-168', 'Buy_Volume_1_-168', 'Buy_Price_2_-168', 'Buy_Volume_2_-168', 'Buy_Price_3_-168', 'Buy_Volume_3_-168', 'Sell_Price_0_-168', 'Sell_Volume_0_-168', 'Sell_Price_1_-168', 'Sell_Volume_1_-168', 'Sell_Price_2_-168', 'Sell_Volume_2_-168', 'Sell_Price_3_-168', 'Sell_Volume_3_-168', 'PeriodExecutedQuantity_-168', 'MeteredMW_-168', 'IDA1PeriodExecutedQuantity_-168', 'IDA2PeriodExecutedQuantity_-168', 'IDA3PeriodExecutedQuantity_-168']
             naive_estimator_columns = ['Buy_Price_0_-48','Buy_Volume_0_-48','Buy_Price_1_-48','Buy_Volume_1_-48','Buy_Price_2_-48','Buy_Volume_2_-48','Buy_Price_3_-48','Buy_Volume_3_-48','Sell_Price_0_-48','Sell_Volume_0_-48','Sell_Price_1_-48','Sell_Volume_1_-48','Sell_Price_2_-48','Sell_Volume_2_-48','Sell_Price_3_-48','Sell_Volume_3_-48']
             model_dictionary = {"ffnn" : False, "svm": False, "rf": True, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": False, "apply_target_scaling":False, "calibration_lookback_days":-75}]        
             model_settings = [{"n_estimators": 100, "max_depth":8, "min_samples_leaf":8}]
             features_to_encode = ['Hour', 'Weekday']
             interaction_terms = []
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')

    approach_1_n_equal_4_rf_b = False
        
    if approach_1_n_equal_4_rf_b:
        name = "approach_1_n_equal_4_rf_b"
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             csv_output_location = raw_csv_location + name
             targets = ['Buy_Price_0','Buy_Volume_0','Buy_Price_1','Buy_Volume_1','Buy_Price_2','Buy_Volume_2','Buy_Price_3','Buy_Volume_3','Sell_Price_0','Sell_Volume_0','Sell_Price_1','Sell_Volume_1','Sell_Price_2','Sell_Volume_2','Sell_Price_3','Sell_Volume_3']
             explanatory_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'Wind-24', 'Wind-48', 'Wind-72', 'Wind-168', 'DemandAggregatedForecast', 'Demand-24', 'Demand-48', 'Demand-72', 'Demand-168', 'MULTI_FUEL_-24', 'GAS_-24', 'COAL_-24', 'MULTI_FUEL_-168', 'GAS_-168', 'COAL_-168', 'InterconnectorNetTotal_-24', 'InterconnectorNetTotal_-168', 'GasPrice', 'GasPrice_-24', 'GasPrice_-168', 'CarbonPrice', 'CarbonPrice_-24', 'CarbonPrice_-168','Buy_Price_0_-48', 'Buy_Volume_0_-48', 'Buy_Price_1_-48', 'Buy_Volume_1_-48', 'Buy_Price_2_-48', 'Buy_Volume_2_-48', 'Buy_Price_3_-48', 'Buy_Volume_3_-48', 'Sell_Price_0_-48', 'Sell_Volume_0_-48', 'Sell_Price_1_-48', 'Sell_Volume_1_-48', 'Sell_Price_2_-48', 'Sell_Volume_2_-48', 'Sell_Price_3_-48', 'Sell_Volume_3_-48', 'PeriodExecutedQuantity_-48', 'MeteredMW_-48', 'IDA1PeriodExecutedQuantity_-48', 'IDA2PeriodExecutedQuantity_-48', 'IDA3PeriodExecutedQuantity_-48', 'Buy_Price_0_-168', 'Buy_Volume_0_-168', 'Buy_Price_1_-168', 'Buy_Volume_1_-168', 'Buy_Price_2_-168', 'Buy_Volume_2_-168', 'Buy_Price_3_-168', 'Buy_Volume_3_-168', 'Sell_Price_0_-168', 'Sell_Volume_0_-168', 'Sell_Price_1_-168', 'Sell_Volume_1_-168', 'Sell_Price_2_-168', 'Sell_Volume_2_-168', 'Sell_Price_3_-168', 'Sell_Volume_3_-168', 'PeriodExecutedQuantity_-168', 'MeteredMW_-168', 'IDA1PeriodExecutedQuantity_-168', 'IDA2PeriodExecutedQuantity_-168', 'IDA3PeriodExecutedQuantity_-168']
             model_dictionary = {"ffnn" : False, "svm": False, "rf": True, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": False, "apply_target_scaling":False, "calibration_lookback_days":-30}]        
             model_settings = [{"n_estimators": 100, "max_depth":8, "min_samples_leaf":8}]
             naive_estimator_columns = ['Buy_Price_0_-48','Buy_Volume_0_-48','Buy_Price_1_-48','Buy_Volume_1_-48','Buy_Price_2_-48','Buy_Volume_2_-48','Buy_Price_3_-48','Buy_Volume_3_-48','Sell_Price_0_-48','Sell_Volume_0_-48','Sell_Price_1_-48','Sell_Volume_1_-48','Sell_Price_2_-48','Sell_Volume_2_-48','Sell_Price_3_-48','Sell_Volume_3_-48']
             features_to_encode = ['Hour', 'Weekday']
             interaction_terms = []
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')

#################################################################################Lasso Forecasts#########################################################################################################################################################################################
                
    #Context: in this section we generate participant forecasts using Lasso modelling approaches.      

    approach_1_n_equal_4_lasso_a = False
        
    if approach_1_n_equal_4_lasso_a:
        name = "approach_1_n_equal_4_lasso_a"
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             csv_output_location = raw_csv_location + name
             targets = ['Buy_Price_0','Buy_Volume_0','Buy_Price_1','Buy_Volume_1','Buy_Price_2','Buy_Volume_2','Buy_Price_3','Buy_Volume_3','Sell_Price_0','Sell_Volume_0','Sell_Price_1','Sell_Volume_1','Sell_Price_2','Sell_Volume_2','Sell_Price_3','Sell_Volume_3']
             explanatory_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'DemandAggregatedForecast', 'GasPrice', 'CarbonPrice', 'Buy_Price_0_-48', 'Buy_Volume_0_-48', 'Buy_Price_1_-48', 'Buy_Volume_1_-48', 'Buy_Price_2_-48', 'Buy_Volume_2_-48', 'Buy_Price_3_-48', 'Buy_Volume_3_-48', 'Sell_Price_0_-48', 'Sell_Volume_0_-48', 'Sell_Price_1_-48', 'Sell_Volume_1_-48', 'Sell_Price_2_-48', 'Sell_Volume_2_-48', 'Sell_Price_3_-48', 'Sell_Volume_3_-48', 'Buy_Price_0_-168', 'Buy_Volume_0_-168', 'Buy_Price_1_-168', 'Buy_Volume_1_-168', 'Buy_Price_2_-168', 'Buy_Volume_2_-168', 'Buy_Price_3_-168', 'Buy_Volume_3_-168', 'Sell_Price_0_-168', 'Sell_Volume_0_-168', 'Sell_Price_1_-168', 'Sell_Volume_1_-168', 'Sell_Price_2_-168', 'Sell_Volume_2_-168', 'Sell_Price_3_-168', 'Sell_Volume_3_-168']
             model_dictionary = {"lasso": True, "ffnn" : False, "svm": False, "rf": False, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": True, "apply_target_scaling":True, "calibration_lookback_days":-120}]        
             model_settings = [{'alpha': 0.1, 'fit_intercept': False}]
             features_to_encode = ['Hour', 'Weekday']
             naive_estimator_columns = ['Buy_Price_0_-48','Buy_Volume_0_-48','Buy_Price_1_-48','Buy_Volume_1_-48','Buy_Price_2_-48','Buy_Volume_2_-48','Buy_Price_3_-48','Buy_Volume_3_-48','Sell_Price_0_-48','Sell_Volume_0_-48','Sell_Price_1_-48','Sell_Volume_1_-48','Sell_Price_2_-48','Sell_Volume_2_-48','Sell_Price_3_-48','Sell_Volume_3_-48']
             interaction_terms = []
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')

    approach_1_n_equal_4_lasso_b = False
        
    if approach_1_n_equal_4_lasso_b:
        name = "approach_1_n_equal_4_lasso_b"
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             csv_output_location = raw_csv_location + name
             targets = ['Buy_Price_0','Buy_Volume_0','Buy_Price_1','Buy_Volume_1','Buy_Price_2','Buy_Volume_2','Buy_Price_3','Buy_Volume_3','Sell_Price_0','Sell_Volume_0','Sell_Price_1','Sell_Volume_1','Sell_Price_2','Sell_Volume_2','Sell_Price_3','Sell_Volume_3']
             explanatory_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'DemandAggregatedForecast', 'GasPrice', 'CarbonPrice', 'Buy_Price_0_-48', 'Buy_Volume_0_-48', 'Buy_Price_1_-48', 'Buy_Volume_1_-48', 'Buy_Price_2_-48', 'Buy_Volume_2_-48', 'Buy_Price_3_-48', 'Buy_Volume_3_-48', 'Sell_Price_0_-48', 'Sell_Volume_0_-48', 'Sell_Price_1_-48', 'Sell_Volume_1_-48', 'Sell_Price_2_-48', 'Sell_Volume_2_-48', 'Sell_Price_3_-48', 'Sell_Volume_3_-48', 'Buy_Price_0_-168', 'Buy_Volume_0_-168', 'Buy_Price_1_-168', 'Buy_Volume_1_-168', 'Buy_Price_2_-168', 'Buy_Volume_2_-168', 'Buy_Price_3_-168', 'Buy_Volume_3_-168', 'Sell_Price_0_-168', 'Sell_Volume_0_-168', 'Sell_Price_1_-168', 'Sell_Volume_1_-168', 'Sell_Price_2_-168', 'Sell_Volume_2_-168', 'Sell_Price_3_-168', 'Sell_Volume_3_-168']
             model_dictionary = {"lasso": True, "ffnn" : False, "svm": False, "rf": False, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": True, "apply_target_scaling":True, "calibration_lookback_days":-30}]        
             model_settings = [{'alpha': 0.1, 'fit_intercept': False}]
             features_to_encode = []
             interaction_terms = []
             naive_estimator_columns = ['Buy_Price_0_-48','Buy_Volume_0_-48','Buy_Price_1_-48','Buy_Volume_1_-48','Buy_Price_2_-48','Buy_Volume_2_-48','Buy_Price_3_-48','Buy_Volume_3_-48','Sell_Price_0_-48','Sell_Volume_0_-48','Sell_Price_1_-48','Sell_Volume_1_-48','Sell_Price_2_-48','Sell_Volume_2_-48','Sell_Price_3_-48','Sell_Volume_3_-48']
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')

#################################################################################SVM Forecasts#########################################################################################################################################################################################

    #Context: in this section we generate participant forecasts using SVM modelling approaches.      

    approach_1_n_equal_4_svm_a = False
        
    if approach_1_n_equal_4_svm_a:
        name = "approach_1_n_equal_4_svm_a"
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             csv_output_location = raw_csv_location + name
             targets = ['Buy_Price_0','Buy_Volume_0','Buy_Price_1','Buy_Volume_1','Buy_Price_2','Buy_Volume_2','Buy_Price_3','Buy_Volume_3','Sell_Price_0','Sell_Volume_0','Sell_Price_1','Sell_Volume_1','Sell_Price_2','Sell_Volume_2','Sell_Price_3','Sell_Volume_3']
             explanatory_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'DemandAggregatedForecast', 'GasPrice', 'CarbonPrice', 'Buy_Price_0_-48', 'Buy_Volume_0_-48', 'Buy_Price_1_-48', 'Buy_Volume_1_-48', 'Buy_Price_2_-48', 'Buy_Volume_2_-48', 'Buy_Price_3_-48', 'Buy_Volume_3_-48', 'Sell_Price_0_-48', 'Sell_Volume_0_-48', 'Sell_Price_1_-48', 'Sell_Volume_1_-48', 'Sell_Price_2_-48', 'Sell_Volume_2_-48', 'Sell_Price_3_-48', 'Sell_Volume_3_-48', 'Buy_Price_0_-168', 'Buy_Volume_0_-168', 'Buy_Price_1_-168', 'Buy_Volume_1_-168', 'Buy_Price_2_-168', 'Buy_Volume_2_-168', 'Buy_Price_3_-168', 'Buy_Volume_3_-168', 'Sell_Price_0_-168', 'Sell_Volume_0_-168', 'Sell_Price_1_-168', 'Sell_Volume_1_-168', 'Sell_Price_2_-168', 'Sell_Volume_2_-168', 'Sell_Price_3_-168', 'Sell_Volume_3_-168']
             model_dictionary = {"lasso": False, "ffnn" : False, "svm": True, "rf": False, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": True, "apply_target_scaling":False, "calibration_lookback_days":-75}]        
             model_settings = [{'C': 1.0, 'epsilon': 0.1}]
             naive_estimator_columns = ['Buy_Price_0_-48','Buy_Volume_0_-48','Buy_Price_1_-48','Buy_Volume_1_-48','Buy_Price_2_-48','Buy_Volume_2_-48','Buy_Price_3_-48','Buy_Volume_3_-48','Sell_Price_0_-48','Sell_Volume_0_-48','Sell_Price_1_-48','Sell_Volume_1_-48','Sell_Price_2_-48','Sell_Volume_2_-48','Sell_Price_3_-48','Sell_Volume_3_-48']
             features_to_encode = []
             interaction_terms = []
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')


    approach_1_n_equal_4_svm_b = False
        
    if approach_1_n_equal_4_svm_b:
        name = "approach_1_n_equal_4_svm_b"
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             csv_output_location = raw_csv_location + name
             targets = ['Buy_Price_0','Buy_Volume_0','Buy_Price_1','Buy_Volume_1','Buy_Price_2','Buy_Volume_2','Buy_Price_3','Buy_Volume_3','Sell_Price_0','Sell_Volume_0','Sell_Price_1','Sell_Volume_1','Sell_Price_2','Sell_Volume_2','Sell_Price_3','Sell_Volume_3']
             explanatory_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'DemandAggregatedForecast', 'GasPrice', 'CarbonPrice', 'Buy_Price_0_-48', 'Buy_Volume_0_-48', 'Buy_Price_1_-48', 'Buy_Volume_1_-48', 'Buy_Price_2_-48', 'Buy_Volume_2_-48', 'Buy_Price_3_-48', 'Buy_Volume_3_-48', 'Sell_Price_0_-48', 'Sell_Volume_0_-48', 'Sell_Price_1_-48', 'Sell_Volume_1_-48', 'Sell_Price_2_-48', 'Sell_Volume_2_-48', 'Sell_Price_3_-48', 'Sell_Volume_3_-48', 'Buy_Price_0_-168', 'Buy_Volume_0_-168', 'Buy_Price_1_-168', 'Buy_Volume_1_-168', 'Buy_Price_2_-168', 'Buy_Volume_2_-168', 'Buy_Price_3_-168', 'Buy_Volume_3_-168', 'Sell_Price_0_-168', 'Sell_Volume_0_-168', 'Sell_Price_1_-168', 'Sell_Volume_1_-168', 'Sell_Price_2_-168', 'Sell_Volume_2_-168', 'Sell_Price_3_-168', 'Sell_Volume_3_-168']
             model_dictionary = {"lasso": False, "ffnn" : False, "svm": True, "rf": False, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": True, "apply_target_scaling":False, "calibration_lookback_days":-120}]        
             model_settings = [{'C': 10.0, 'epsilon': 0.1}]
             features_to_encode = []
             interaction_terms = []
             naive_estimator_columns = ['Buy_Price_0_-48','Buy_Volume_0_-48','Buy_Price_1_-48','Buy_Volume_1_-48','Buy_Price_2_-48','Buy_Volume_2_-48','Buy_Price_3_-48','Buy_Volume_3_-48','Sell_Price_0_-48','Sell_Volume_0_-48','Sell_Price_1_-48','Sell_Volume_1_-48','Sell_Price_2_-48','Sell_Volume_2_-48','Sell_Price_3_-48','Sell_Volume_3_-48']
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')

#################################################################################FFNN Forecasts#########################################################################################################################################################################################

    #Context: in this section we generate participant forecasts using a FFNN modelling approach.      

    approach_1_n_equal_4_ffnn = False
        
    if approach_1_n_equal_4_ffnn:
        name = "approach_1_n_equal_4_ffn"               
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             csv_output_location = raw_csv_location + name
             targets = ['Buy_Price_0','Buy_Volume_0','Buy_Price_1','Buy_Volume_1','Buy_Price_2','Buy_Volume_2','Buy_Price_3','Buy_Volume_3','Sell_Price_0','Sell_Volume_0','Sell_Price_1','Sell_Volume_1','Sell_Price_2','Sell_Volume_2','Sell_Price_3','Sell_Volume_3']
             explanatory_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'DemandAggregatedForecast', 'GasPrice', 'CarbonPrice', 'Buy_Price_0_-48', 'Buy_Volume_0_-48', 'Buy_Price_1_-48', 'Buy_Volume_1_-48', 'Buy_Price_2_-48', 'Buy_Volume_2_-48', 'Buy_Price_3_-48', 'Buy_Volume_3_-48', 'Sell_Price_0_-48', 'Sell_Volume_0_-48', 'Sell_Price_1_-48', 'Sell_Volume_1_-48', 'Sell_Price_2_-48', 'Sell_Volume_2_-48', 'Sell_Price_3_-48', 'Sell_Volume_3_-48', 'Buy_Price_0_-168', 'Buy_Volume_0_-168', 'Buy_Price_1_-168', 'Buy_Volume_1_-168', 'Buy_Price_2_-168', 'Buy_Volume_2_-168', 'Buy_Price_3_-168', 'Buy_Volume_3_-168', 'Sell_Price_0_-168', 'Sell_Volume_0_-168', 'Sell_Price_1_-168', 'Sell_Volume_1_-168', 'Sell_Price_2_-168', 'Sell_Volume_2_-168', 'Sell_Price_3_-168', 'Sell_Volume_3_-168']
             model_dictionary = {"lasso": False, "ffnn" : True, "svm": False, "rf": False, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": True, "apply_target_scaling":True, "calibration_lookback_days":-120}]        
             model_settings = [{'m_epochs': 150, 'number_of_nodes_list': [64, 64], 'keep_epochs_unchanged': True, 'drop_out': 0.2, 'learning_rate': 0.01}]
             naive_estimator_columns = ['Buy_Price_0_-48','Buy_Volume_0_-48','Buy_Price_1_-48','Buy_Volume_1_-48','Buy_Price_2_-48','Buy_Volume_2_-48','Buy_Price_3_-48','Buy_Volume_3_-48','Sell_Price_0_-48','Sell_Volume_0_-48','Sell_Price_1_-48','Sell_Volume_1_-48','Sell_Price_2_-48','Sell_Volume_2_-48','Sell_Price_3_-48','Sell_Volume_3_-48']
             features_to_encode = []
             interaction_terms = []
             earlystopping=True 
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points,earlystopping=earlystopping)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')

#################################################################################Approach 1, n = 5, forecasts#########################################################################################################################################################################################

    #Context: in this section using Approach 1 with n = 5, we use a lasso modelling approach to generate forecasts.      

    approach_1_n_equal_5_lasso = False
        
    if approach_1_n_equal_5_lasso:
        name = "approach_1_n_equal_5_lasso"
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             csv_output_location = raw_csv_location + name
             naive_estimator_columns = ['Buy_Price_0_-48','Buy_Volume_0_-48','Buy_Price_1_-48','Buy_Volume_1_-48','Buy_Price_2_-48','Buy_Volume_2_-48','Buy_Price_3_-48','Buy_Volume_3_-48','Buy_Price_4_-48','Buy_Volume_4_-48','Sell_Price_0_-48','Sell_Volume_0_-48','Sell_Price_1_-48','Sell_Volume_1_-48','Sell_Price_2_-48','Sell_Volume_2_-48','Sell_Price_3_-48','Sell_Volume_3_-48','Sell_Price_4_-48','Sell_Volume_4_-48']
             targets = ['Buy_Price_0','Buy_Volume_0','Buy_Price_1','Buy_Volume_1','Buy_Price_2','Buy_Volume_2','Buy_Price_3','Buy_Volume_3','Buy_Price_4','Buy_Volume_4','Sell_Price_0','Sell_Volume_0','Sell_Price_1','Sell_Volume_1','Sell_Price_2','Sell_Volume_2','Sell_Price_3','Sell_Volume_3','Sell_Price_4','Sell_Volume_4']
             explanatory_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'DemandAggregatedForecast', 'GasPrice', 'CarbonPrice', 'Buy_Price_0_-48', 'Buy_Volume_0_-48', 'Buy_Price_1_-48', 'Buy_Volume_1_-48', 'Buy_Price_2_-48', 'Buy_Volume_2_-48', 'Buy_Price_3_-48', 'Buy_Volume_3_-48', 'Buy_Price_4_-48', 'Buy_Volume_4_-48', 'Sell_Price_0_-48', 'Sell_Volume_0_-48', 'Sell_Price_1_-48', 'Sell_Volume_1_-48', 'Sell_Price_2_-48', 'Sell_Volume_2_-48', 'Sell_Price_3_-48', 'Sell_Volume_3_-48', 'Sell_Price_4_-48', 'Sell_Volume_4_-48', 'Buy_Price_0_-168', 'Buy_Volume_0_-168', 'Buy_Price_1_-168', 'Buy_Volume_1_-168', 'Buy_Price_2_-168', 'Buy_Volume_2_-168', 'Buy_Price_3_-168', 'Buy_Volume_3_-168', 'Buy_Price_4_-168', 'Buy_Volume_4_-168', 'Sell_Price_0_-168', 'Sell_Volume_0_-168', 'Sell_Price_1_-168', 'Sell_Volume_1_-168', 'Sell_Price_2_-168', 'Sell_Volume_2_-168', 'Sell_Price_3_-168', 'Sell_Volume_3_-168', 'Sell_Price_4_-168', 'Sell_Volume_4_-168']
             model_dictionary = {"lasso": True, "ffnn" : False, "svm": False, "rf": False, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": True, "apply_target_scaling":True, "calibration_lookback_days":-30}]        
             model_settings = [{'alpha': 0.1, 'fit_intercept': False}]
             features_to_encode = []
             interaction_terms = []
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')

#################################################################################Approach 2 forecasts#########################################################################################################################################################################################

    #Context: in this section using Approach 2, with varying dimensions, we generate forecasts using different modelling approaches. 

    approach_2_dim_equal_25_lasso = False
        
    if approach_2_dim_equal_25_lasso:
        name = "approach_2_dim_equal_25_lasso"
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             csv_output_location = raw_csv_location + name
             quantile_columns = ["Minimum","Lower"] + ["Quantile_"+str(i) for i in range(0,105,5)] + ["Upper","Maximum"]
             targets = [x for x in quantile_columns]
             naive_estimator_columns = [x+"_-48" for x in quantile_columns]
             naive_estimator_columns_lag_168 = [x+"_-168" for x in quantile_columns]
             temp_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'DemandAggregatedForecast', 'GasPrice', 'CarbonPrice']
             explanatory_variables = temp_variables + naive_estimator_columns + naive_estimator_columns_lag_168
             model_dictionary = {"lasso": True, "ffnn" : False, "svm": False, "rf": False, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": True, "apply_target_scaling":True, "calibration_lookback_days":-30}]        
             model_settings = [{'alpha': 0.1, 'fit_intercept': False}]
             features_to_encode = []
             interaction_terms = []
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')

    approach_2_dim_equal_15_lasso = False
        
    if approach_2_dim_equal_15_lasso:
        name = "approach_2_dim_equal_15_lasso"
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             quantile_columns = ["Minimum","Lower"] + ["Quantile_"+str(i) for i in range(0,110,10)] + ["Upper","Maximum"]
             targets = [x for x in quantile_columns]
             naive_estimator_columns = [x+"_-48" for x in quantile_columns]
             naive_estimator_columns_lag_168 = [x+"_-168" for x in quantile_columns]
             temp_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'DemandAggregatedForecast', 'GasPrice', 'CarbonPrice']
             explanatory_variables = temp_variables + naive_estimator_columns + naive_estimator_columns_lag_168
             csv_output_location = raw_csv_location + name
             model_dictionary = {"lasso": True, "ffnn" : False, "svm": False, "rf": False, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": True, "apply_target_scaling":True, "calibration_lookback_days":-30}]        
             model_settings = [{'alpha': 0.1, 'fit_intercept': False}]
             features_to_encode = []
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')

    approach_2_dim_equal_25_svm = False
        
    if approach_2_dim_equal_25_svm:
        name = "approach_2_dim_equal_25_svm"
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             csv_output_location = raw_csv_location + name
             quantile_columns = ["Minimum","Lower"] + ["Quantile_"+str(i) for i in range(0,105,5)] + ["Upper","Maximum"]
             targets = [x for x in quantile_columns]
             naive_estimator_columns = [x+"_-48" for x in quantile_columns]
             naive_estimator_columns_lag_168 = [x+"_-168" for x in quantile_columns]
             temp_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'DemandAggregatedForecast', 'GasPrice', 'CarbonPrice']
             explanatory_variables = temp_variables + naive_estimator_columns + naive_estimator_columns_lag_168
             model_dictionary = {"lasso": False, "ffnn" : False, "svm": True, "rf": False, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": True, "apply_target_scaling":False, "calibration_lookback_days":-120}]        
             model_settings = [{'C': 10.0, 'epsilon': 0.1}]
             features_to_encode = []
             interaction_terms = []
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')

#################################################################################Approach 1, n = 4, but we model the buy and sell components separately#########################################################################################################################################################################################

    #Context: in this section using Approach 1 with n = 4, we model the buy and sell components separately, using a Lasso modelling approach, 

    approach_1_n_equal_4_lasso_buy = False
        
    if approach_1_n_equal_4_lasso_buy:
        name = "approach_1_n_equal_4_lasso_buy"
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             csv_output_location = raw_csv_location + name
             targets = ['Buy_Price_0','Buy_Volume_0','Buy_Price_1','Buy_Volume_1','Buy_Price_2','Buy_Volume_2','Buy_Price_3','Buy_Volume_3']
             explanatory_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'DemandAggregatedForecast', 'GasPrice', 'CarbonPrice', 'Buy_Price_0_-48', 'Buy_Volume_0_-48', 'Buy_Price_1_-48', 'Buy_Volume_1_-48', 'Buy_Price_2_-48', 'Buy_Volume_2_-48', 'Buy_Price_3_-48', 'Buy_Volume_3_-48', 'Sell_Price_0_-48', 'Sell_Volume_0_-48', 'Sell_Price_1_-48', 'Sell_Volume_1_-48', 'Sell_Price_2_-48', 'Sell_Volume_2_-48', 'Sell_Price_3_-48', 'Sell_Volume_3_-48', 'Buy_Price_0_-168', 'Buy_Volume_0_-168', 'Buy_Price_1_-168', 'Buy_Volume_1_-168', 'Buy_Price_2_-168', 'Buy_Volume_2_-168', 'Buy_Price_3_-168', 'Buy_Volume_3_-168', 'Sell_Price_0_-168', 'Sell_Volume_0_-168', 'Sell_Price_1_-168', 'Sell_Volume_1_-168', 'Sell_Price_2_-168', 'Sell_Volume_2_-168', 'Sell_Price_3_-168', 'Sell_Volume_3_-168']
             model_dictionary = {"lasso": True, "ffnn" : False, "svm": False, "rf": False, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": True, "apply_target_scaling":True, "calibration_lookback_days":-30}]        
             model_settings = [{'alpha': 0.1, 'fit_intercept': False}]
             features_to_encode = []
             interaction_terms = []
             naive_estimator_columns = ['Buy_Price_0_-48','Buy_Volume_0_-48','Buy_Price_1_-48','Buy_Volume_1_-48','Buy_Price_2_-48','Buy_Volume_2_-48','Buy_Price_3_-48','Buy_Volume_3_-48']
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')

    approach_1_n_equal_4_lasso_sell = False
        
    if approach_1_n_equal_4_lasso_sell:
        name = "approach_1_n_equal_4_lasso_sell"
        for unit in unit_names:
             raw_csv_location = "INSERT INFO HERE"
             targets = ['Sell_Price_0','Sell_Volume_0','Sell_Price_1','Sell_Volume_1','Sell_Price_2','Sell_Volume_2','Sell_Price_3','Sell_Volume_3']
             explanatory_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'DemandAggregatedForecast', 'GasPrice', 'CarbonPrice', 'Buy_Price_0_-48', 'Buy_Volume_0_-48', 'Buy_Price_1_-48', 'Buy_Volume_1_-48', 'Buy_Price_2_-48', 'Buy_Volume_2_-48', 'Buy_Price_3_-48', 'Buy_Volume_3_-48', 'Sell_Price_0_-48', 'Sell_Volume_0_-48', 'Sell_Price_1_-48', 'Sell_Volume_1_-48', 'Sell_Price_2_-48', 'Sell_Volume_2_-48', 'Sell_Price_3_-48', 'Sell_Volume_3_-48', 'Buy_Price_0_-168', 'Buy_Volume_0_-168', 'Buy_Price_1_-168', 'Buy_Volume_1_-168', 'Buy_Price_2_-168', 'Buy_Volume_2_-168', 'Buy_Price_3_-168', 'Buy_Volume_3_-168', 'Sell_Price_0_-168', 'Sell_Volume_0_-168', 'Sell_Price_1_-168', 'Sell_Volume_1_-168', 'Sell_Price_2_-168', 'Sell_Volume_2_-168', 'Sell_Price_3_-168', 'Sell_Volume_3_-168']
             csv_output_location = raw_csv_location + name
             model_dictionary = {"lasso": True, "ffnn" : False, "svm": False, "rf": False, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": True, "apply_target_scaling":True, "calibration_lookback_days":-30}]        
             model_settings = [{'alpha': 0.1, 'fit_intercept': False}]
             naive_estimator_columns = ['Sell_Price_0_-48','Sell_Volume_0_-48','Sell_Price_1_-48','Sell_Volume_1_-48','Sell_Price_2_-48','Sell_Volume_2_-48','Sell_Price_3_-48','Sell_Volume_3_-48']
             features_to_encode = []
             interaction_terms = []
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')

#################################################################################Approach 2, dim = 25, modelling the speculators as combined group#########################################################################################################################################################################################

    #Context: in this section using Approach 2 with dim = 25, using a Lasso modelling approach, we model the speculators as a single entity.
    #         We forecast the aggregate buys, and the aggregate sells, separately.

    approach_2_dim_equal_25_lasso_speculators = False
        
    if approach_2_dim_equal_25_lasso_speculators:
        name = "approach_2_dim_equal_25_lasso_speculators"
        unit_names = ['buy speculator','sell speculator']
        head = ['Buy','Sell']
        for i in range(0,len(unit_names)):
             quantile_columns = ["Minimum","Lower"] + ["Quantile_"+str(i) for i in range(0,105,5)] + ["Upper","Maximum"]
             raw_csv_location = "INSERT INFO HERE"
             csv_output_location = raw_csv_location + name
             unit =  unit_names[i]
             t = head[i]
             quantile_columns = [ t + "_" + x for x in quantile_columns]
             targets = [x for x in quantile_columns]
             naive_estimator_columns = [x+"_-48" for x in quantile_columns] + ['PEQ_Buy_-48','PEQ_Sell_-48']
             naive_estimator_columns_lag_168 = [x+"_-168" for x in quantile_columns] + ['PEQ_Buy_-168','PEQ_Sell_-168']
             temp_variables = ['-24', '-48', '-72', '-168', 'BM-24', 'BM-48', 'BM-72', 'BM-168', 'WindAggregatedForecast', 'DemandAggregatedForecast', 'GasPrice', 'CarbonPrice']
             explanatory_variables = temp_variables + naive_estimator_columns + naive_estimator_columns_lag_168
             model_dictionary = {"lasso": True, "ffnn" : False, "svm": False, "rf": False, "knn" : False, "naive_estimator_2day_lookback":False,"naive_estimator_7day_lookback":False}             
             meta_settings = [{"scale_explanatory_variables": True, "apply_target_scaling":True, "calibration_lookback_days":-30}]        
             model_settings = [{'alpha': 0.1, 'fit_intercept': False}]
             features_to_encode = []
             interaction_terms = []
             error_messages, info_messages = multitarget_walk_forward_validation_wrapper(db_table_name=name, unit_names=[unit], raw_csv_location=raw_csv_location,\
                                                                                         model_dictionary=model_dictionary, targets=targets, explanatory_variables=explanatory_variables,interaction_terms=interaction_terms,\
                                                                                         features_to_encode=features_to_encode, meta_settings=meta_settings, model_settings=model_settings, start_time=start_time,\
                                                                                         end_time=end_time,naive_estimator_columns=naive_estimator_columns,dataset_type="_basic_ts_dataset",save_summary_to_db=save_summary_to_db,save_test_to_csv=save_test_to_csv,\
                                                                                         csv_output_location=csv_output_location,plot=plot,clock_change_start=clock_change_start,clock_change_end=clock_change_end,\
                                                                                         min_number_of_calibration_points=min_number_of_calibration_points)
             logtime = [datetime.now()]
             with open(csv_output_location+"\\error_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in error_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')
             with open(csv_output_location+"\\info_messages.csv", 'a', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(logtime)
                 for item in info_messages:
                     writer.writerow([item,])
                 writer.writerow('\n')


