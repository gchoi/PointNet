'''
@Description: Module for PyTorch logging.
@Developed by: Alex Choi
@Date: Aug. 2, 2022
@Contact: cinema4dr12@gmail.com
'''

#%% Importing packages
import logging


#%% Logging
level = logging.INFO
logging.basicConfig(level=level, format='%(asctime)s - [%(levelname)s] - %(message)s')