# Code for Winberry 2018 replication

Two ways to replicate:

### The easy way:
Please run the binder link below and open the ipython notebook, go to cells and hit run all.  This will reproduce everything but the final IRF graphs from Matlab, which is due to an incompatibility between the version of Octave on Ubuntu and Dynare. It will, however, print out the numerical IRF estimated and correlation coefficient.

### The harder 100% way:

Please run the `replication_winberry_2018.ipynb` notebook in Jupyter. You can install Jupyter by downloading the anaconda package from python. After `cd`-ing to the directory containing this file run `jupyter lab replication_winberry_2018.ipynb`.
To have this file work you need the following packages installed (from conda):
- numpy 
- scipy
- matplotlib
- numba

For an easy way to do this and to avoid interfering with other files I have included an environment.yml file in the .zip file. To create a new conda environment including everything needed for running (outside the matlab package for python.) To create this new environment after having installed anaconda python open a Terminal and run:

```
cd \directory_location\unzipped_replication_file_folder
conda env create -f environment.yml
conda activate winberry-replication
```
This will setup the winberry-replication environment needed to run the program. 
If you would later like to remove this environment run:
``` 
conda remove --name winberry-replication --all
```


Finally, you will need to install the matlab package for python. To install this, please install a version of Matlab with Dynare installed and linked (following the Dynare instructions page.) To avoid any issues please make sure the newest version of Dynare is installed. Then please follow the guide listed here to install the python package for Matlab under the 'Install Engine API' header: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

Please make sure you have run ```conda activate winberry-replication``` before running the Matlab ```setup.py``` installer. 

It should take less than a minute to install. 

Finally, after activating the environment and installing matlab for python you can run: 
```
jupyter lab directory_to_unzipped_file\replication_winberry_2018.ipynb
```


Backend files written to run the IPython Notebook are in SteadyState_Libraries. Files written to run Dynare are in Dynare_Files.
You can also run this directly from the Binder link below. 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bkaplowitz/Code-Replication-Winberry/HEAD)


