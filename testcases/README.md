## Running IMAT Tests
This document explains how to run the test cases for the Interactive Melody Annotation Tool (IMAT).

### Setup Test Data

Place test audio files in the ```testcases/test_data/audio/``` directory
Place corresponding ground truth CSV files in ```testcases/test_data/ground_truth/```

CSV files must have the same base name as their audio counterparts
Example: ```audio/song1.wav``` → ```ground_truth/song1.csv```



### Running the Tests

Execute the full test suite:

- Change the directory path: ```cd testcases/```
- Run the file: ```python3 test_cases.py```
