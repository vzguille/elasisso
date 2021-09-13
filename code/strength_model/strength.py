# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:50:26 2020

@author: richardcouperthwaite
"""

import numpy as np
# from tc_python import TCPython, ThermodynamicQuantity
#import json
import pandas as pd


def model_Curtin(elements, alpha=1/12, bar_C=None, prop_dfs = []):
    if not prop_dfs:
        elast_const = pd.read_csv('strength_model/VOLUMEdata/ULTIMATEElasticConstants.csv')
        volumes = pd.read_csv('strength_model/VOLUMEdata/ULTIMATEVolumes.csv').to_dict(orient = 'list')
    else:
        elast_const = prop_dfs[0]
        volumes = prop_dfs[1]
    if bar_C is None:
        bar_C11 = 0
        bar_C12 = 0
        bar_C44 = 0  
        for ele in elements:
            bar_C11 += elast_const[ele][0]*elements[ele]['fraction']
            bar_C12 += elast_const[ele][1]*elements[ele]['fraction']
            bar_C44 += elast_const[ele][2]*elements[ele]['fraction']
    else:
        bar_C11,bar_C12,bar_C44=bar_C[0],bar_C[1],bar_C[2]
    volumes = {key : volumes[key][0] for key in volumes.keys()}
    
    bar_V = 0
    
    phases_record = []
    el_for_total = []
    fr_for_total = []
    
    for ele in elements:
        el_for_total.append(ele)
        fr_for_total.append(elements[ele]['fraction'])
        
        elements[ele]['BCCVol'] = volumes[ele]
        
        bar_V += elements[ele]['BCCVol']*elements[ele]['fraction']
    #print('elas',bar_C11,bar_C12,bar_C44)

    misfit_Vol_Factor = 0
    
    misfit = {}
    
    for ele in elements:
        misfit[ele] = (elements[ele]['BCCVol']-bar_V)
        misfit_Vol_Factor += elements[ele]['fraction']*((elements[ele]['BCCVol']-bar_V)**2)

    mu_bar = np.sqrt(0.5*(bar_C44)*(bar_C11-bar_C12))
    B_bar = (bar_C11 + 2*bar_C12)/3
    nu_bar = (3*B_bar-2*mu_bar)/(2*(3*B_bar+mu_bar))
    #    nu_bar = (3*B_bar-2*mu_bar)/(2*(3*B_bar+2*mu_bar))

    unitCell_a = (2*bar_V)**(1/3)#(((3*bar_V)/(4*np.pi))**(1/3))*(4/np.sqrt(3))
    
    b_bar = (unitCell_a*np.sqrt(3)/2)# (np.sqrt(3)/2)*(bar_V**(1/3))*2
    
    tau_y_zero = 0.040*(alpha**(-1/3))*mu_bar*(((1+nu_bar)/(1-nu_bar))**(4/3))*(((misfit_Vol_Factor)/(b_bar**6))**(2/3))
    
    delta_E_b = 2.00*(alpha**(1/3))*mu_bar*(b_bar**3)*(((1+nu_bar)/(1-nu_bar))**(2/3))*(((misfit_Vol_Factor)/(b_bar**6))**(1/3))

    output = {'tau_y_0': tau_y_zero,
              'delta_Eb': delta_E_b/160.2176621,
              'Average C': [bar_C11, bar_C12, bar_C44],
              'misfit': misfit,
              'misfit_Vol_Factor':misfit_Vol_Factor,
              'scale_misfit':(((misfit_Vol_Factor)/(b_bar**6))**(2/3)),
              'a': unitCell_a,
              'bar_V': bar_V,
              'phases_record': phases_record,
              'b_bar': b_bar,
              'mu_bar': mu_bar,
              'nu_bar': nu_bar}   
    return output

def temp_model(results, eps_dot, approx_model=False, T = 2273):
    eps_dot_0 = 1e4
    k = 8.617e-5
    if approx_model:
        tau = results['tau_y_0']*np.exp(-(1/0.55)*((((k*T)/(results['delta_Eb']))*np.log(eps_dot_0/eps_dot))**(0.91)))
    else:
        tau_low = results['tau_y_0']*(1-(((k*T)/(results['delta_Eb']))*np.log(eps_dot_0/eps_dot))**(2/3))
        tau_high = results['tau_y_0']*np.exp(-(1/0.55)*((k*T)/(results['delta_Eb']))*np.log(eps_dot_0/eps_dot))
        tau = []
        for i in range(len(T)):
            if tau_low[i]/results['tau_y_0'] > 0.5:
                tau.append(tau_low[i])
            else:
                tau.append(tau_high[i])
        tau = np.array(tau)
    return tau

def model_Control(element_list, comp_list, bar_C=None, T = 2273, prop_dfs = []):
    model_input = {}
    for i in range(len(element_list)):
        model_input[element_list[i]] = {'fraction': comp_list[i]}

    
    
    if bar_C is None:
        result = model_Curtin(model_input) 
    else:
        result = model_Curtin(model_input,bar_C=bar_C) 
    
    out = temp_model(result, 0.001, True, T)
    return (result['tau_y_0'], result['delta_Eb'], out)

def model_Control_all(element_list, comp_list, bar_C=None, T = 2273, prop_dfs = []):
    model_input = {}
    for i in range(len(element_list)):
        model_input[element_list[i]] = {'fraction': comp_list[i]}

    
    
    if bar_C is None:
        result = model_Curtin(model_input) 
    else:
        result = model_Curtin(model_input,bar_C=bar_C) 
    return result

if __name__ == "__main__":
    elements = ['Mo', 'Nb', 'Ta', 'V', 'W']    
    comp = [0.217,0.206,0.156,0.21,0.211]
    
    out = model_Control(elements, comp)
    
    print(out)
    