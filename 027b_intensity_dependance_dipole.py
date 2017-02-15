# Broken script
import numpy as np
import matplotlib.pyplot as plt

import GasFlowHLCalculator.qbs_fill as qf



fills = [5219, 5222, 5223]

fill_qbs_dict = {}
for fill in fills:
    qbs_ob = qf.compute_qbs_special(fill)
    fill_qbs_dict[fill] = qbs_ob
