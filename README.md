# Complex Systems and Networks Github Repository for Spring Semester 2021

Installation: 
```pip install -r requirements.txt``` 

Please read the included document ```something.pdf``` for background on the project and a description of the goals this
software is attempting to accomplish.

This codebase contains the software developed for the EECE7065 course at the University of Cincinnati during spring
semester 2021. Both the code for the final project and Homework 2 are included.

Project Usage:

All executable files for data collection in this project are located in the ```tests``` directory. In this directory the 
file ```gol_optimization_template.py``` provides an example of how to setup a single run of the GA. Options
are provided to specify the parameters of interest as well as the fitness function. 

Additionally, the file ```multi_test.py``` provides an example of how to set up multiple runs of the GA for a single
parameter of interest.

After each run of the GA, the parameters for that run, along with the best found chromosome are logged in the 
```tests/test_logs``` directory. The parameter file has an ```.xlsx``` extension and the chromosome has a ```.txt```
extension with the same prefix as the ```.py``` file.

To play back the animation of a chromosome, modify the file ```tests/view_successful_configs.py```. This file has an
option to modify the file prefix for a test and loads in the chromosome for an animation.

## Project Contributors
Vita Borovyk - borovyva@ucmail.uc.edu

Nicholas DeGroote - degroona@mail.uc.edu

Lynn Pickering - pickerln@mail.uc.edu

Owen Traubert - traubeod@mail.uc.edu
