import pandas as pd
import numpy as np


def overall_analysis(accs, aucs, sims):
    accs = np.array(accs)
    aucs = np.array(aucs)
    
    sharpe_ratio = np.array([sim['Sharpe Ratio'] for sim in sims])
    sortino_ratio = np.array([sim['Sortino Ratio'] for sim in sims])
    calmar_ratio = np.array([sim['Calmar Ratio'] for sim in sims])
    ann_return = np.array([sim['Return (Ann.) [%]'] for sim in sims])
    
    return np.mean(accs), np.mean(aucs), np.mean(sharpe_ratio), np.mean(sortino_ratio), np.mean(calmar_ratio), np.mean(ann_return)