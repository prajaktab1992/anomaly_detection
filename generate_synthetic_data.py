# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:31:49 2020

@author: PrajaktaB2
"""
import tsaug
import pandas as pd
import numpy as np

def generate_data(col_name):
    ts_data = pd.read_csv("Fig8.csv")
    ts_data = ts_data[['angle', 'current', 'fkm4tate', 'level',
       'load', 'pressure', 'temperature', 'waterFraction ',"label"]];
                       
    angle = ts_data["angle"].values
    current = ts_data["current"].values
    fkm4tate = ts_data["fkm4tate"].values
    level = ts_data["level"].values
    load = ts_data["load"].values
    pressure = ts_data["pressure"].values
    temperature =  ts_data["temperature"].values
    waterFraction = ts_data["waterFraction "].values
    
    angle_list =[]
    current_list =[]
    fkm4tate_list=[]
    level_list=[]
    load_list = []
    pressure_list = []
    temp_list = []
    waterFraction_list = []
    for i in range(1,20):
       
        randInt =int(np.random.randint(min(range(1,9)),max(range(1,9)),1)[0])
        
        current_aug, angle_aug= tsaug.TimeWarp(n_speed_change=randInt, max_speed_ratio=3).augment(current,angle)
        current_aug, angle_aug  = tsaug.Drift(max_drift=0.7, n_drift_points=randInt).augment(current_aug, angle_aug )
        
        level_aug,fkm4tate_aug= tsaug.TimeWarp(n_speed_change=randInt, max_speed_ratio=3).augment(level,fkm4tate)
        level_aug,fkm4tate_aug  = tsaug.Drift(max_drift=0.7, n_drift_points=randInt).augment(level_aug,fkm4tate_aug) 
        
        pressure_aug,load_aug = tsaug.TimeWarp(n_speed_change=randInt, max_speed_ratio=3).augment(pressure,load)
        pressure_aug,load_aug  = tsaug.Drift(max_drift=0.7, n_drift_points=randInt).augment(pressure_aug,load_aug )
        
        temperature_aug,waterFraction_aug = tsaug.TimeWarp(n_speed_change=randInt, max_speed_ratio=3).augment(temperature,waterFraction)
        temperature_aug,waterFraction_aug  = tsaug.Drift(max_drift=0.7, n_drift_points=randInt).augment(temperature_aug,waterFraction_aug )
        
        angle_list =angle_list + list (angle_aug)
        current_list =current_list + list (current_aug)
        fkm4tate_list=fkm4tate_list + list (fkm4tate_aug)
        level_list=level_list + list (level_aug)
        load_list = load_list + list (load_aug)
        pressure_list = pressure_list + list (pressure_aug)
        temp_list =temp_list + list (temperature_aug)
        waterFraction_list = waterFraction_list + list (waterFraction_aug)
    
    if col_name=="temperature":
        return temp_list
    