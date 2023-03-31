
import time
import argparse
import os
import datetime
import time
from pathlib import Path
import numpy as np
import pandas as pd
from jtop import jtop

def get_jetson_information()-> dict:
    """
    This function returns a dictionary containing information about the Jetson device. 
    It uses the `jtop` library to interact with the device and retrieve information such as the current nvpmodel, 
    whether jetson_clocks is activated, the module name and L4T version. 
    The function first checks if jetson_clocks is activated and if the nvpmodel is set to 'MAXN' for maximum power. 
    If not, it activates jetson_clocks and sets the nvpmodel to 'MAXN'. 
    Then it retrieves the information and returns it in a dictionary.
    """
    jetson_information = {}
    # try:
    with jtop() as jetson:
        # jetson.ok() will provide the proper update frequency
        while jetson.ok():
            # Activate jetson_clocks if not already activated
            if not jetson.jetson_clocks:
                jetson.jetson_clocks = True
                print("jetson_clocks activated")
            # Set nvpmodel to 'MAXN' for maximum power if not already set
            if not jetson.nvpmodel == 'MAXN' :   # Run with max power
                jetson.nvpmodel = 0
                print(f'Set nvpmodel to "MAXN" power')
            time.sleep(1)
            # Store the information in a dictionary
            jetson_information["Nvpmodel"] = jetson.nvpmodel
            jetson_information["Jetson_Clocks"] = jetson.jetson_clocks 
            jetson_information["Module"] = jetson.board['hardware']['Module']
            jetson_information["L4T"] = jetson.board['hardware']['L4T']
            break
    print("Jetson information", jetson_information)
    return jetson_information

def save_summary(save_data: dict , path_save:str , **kwds):
    """
    This function saves a summary of data to a CSV file. 
    It takes in a dictionary of data to be saved and the path to the file where the data should be saved. 
    If the file already exists, the new data is appended to the existing data. 
    If the file does not exist, a new file is created and the data is saved.
    """
    # Convert the dictionary of data to a DataFrame
    dnew = pd.DataFrame(save_data, index=[0])
    # Check if the file already exists
    if os.path.exists(path_save):
        dori = pd.read_csv(path_save)
        dori = pd.concat([dori, dnew])
        dori.to_csv(path_save, index=False)
    else:
        dnew.to_csv(path_save, index=False)
    print(f'\n{path_save} saved!')
    
