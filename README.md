# IMAT - Interactive Melody Annotation Tool

## Overview
IMAT is a tool designed to correct singing melodies obtained from machine-learning melody estimation algorithms. 


## Usage
How to setup the tool?

**Prerequisites**
- Clone this repository on your local system and setup environment
  ```
      git clone https://github.com/madhavlab/imat_taslp.git
      cd imat_taslp
      python3 requirement.txt
  ```

## Tool Manual
- On the terminal, activate the new environment and run the Python script - 'app.py'.
- After executing 'app.py', copy the link that appears on the terminal and open it on the browser, or press Ctrl+link to open it directly on the browser. The link should look like this - <ins>Running on</ins> http://127.0.0.1:3000
  
- **STEP 1:** Click on the 'Upload audio' button and browse through your system to annotate the audio you wish. Please make sure you upload a '.wav' audio file. The maximum duration allowed for the audio is 60 seconds.
- **STEP 2:** Once the audio is uploaded, click on the 'Show Spectrogram' button to visualize the spectrogram.
- **STEP 3:** To visualize the estimated melody, click on the 'Show Melody' checkbox. The spectrogram overlayed with the estimated melody appears.
- **STEP 4:** The user can correct the melody as follows:
1. The user visually observes the region where the estimated melody is incorrect and selects a particular range of the spectrogram to annotate the incorrect melody. Once selected, the user clicks on `Annotate'.
2. After clicking on 'Annotate', another interactive plot appears, which contains the estimated melody (at 10ms hop size) in that range. The melody at each time frame is an anchor point that can modified by the user. Another bar plot appears just below the interactive plot that indicates confidence (intensity of the color increases from the lowest to highest confidence values) at each time frame in that range.
3. The user can visually observe the low-confidence frames and manually correct the melody in those frames. The user can either drag a single melody anchor point or select consecutive anchor points (by clicking and dragging a rectangle over the anchor points).
4. The user first corrects the incorrect melody anchor points corresponding to the low-confidence frames by either dragging them to align with the F0 contour in the spectrogram or clicking on `Remove Pitches' if no melody is present. The user can auditorily validate the annotation. **Don't forget to right-click on the interactive plot after performing actions on the consecutively selected anchor points, as it deselects the selected range of anchor points.**
5. Repeat the process from 1-4 until the user is satisfied by annotating **only** the low-confident frames of the entire audio.
6. After this, the user clicks on the 'Retrain Model' button to adapt the model to these corrections. This is referred to as the *s=1* iteration of Adaptive Annotation.
7. The retrained model predicts an updated estimated melody, and the user can visualize it as in STEP 3. Also, the confidence values at each time frame are also updated.
9. The points from 1 to 7 are repeated for *s* iterations of Adaptive Annotation until an overall accuracy of at least 95%.
10. To further improve the overall accuracy, the user always has the option of correcting the remaining melody manually.
- **STEP 5:** Once the user is satisfied with the corrections, the annotations can be downloaded by clicking on the 'Download CSV' button.

**Follow the same procedure to annotate any polyphonic audio.**

## Citation
If you use IMAT for annotating the polyphonic audios, please cite us
```
  *citation*
```
Also, if you use melody annotations of a subset of the FMA-small dataset, please cite the dataset 
```
  *citation*
```

## Contact
For any questions or feedback, please contact:
- Kavya Ranjan Saxena (kavyars@iitk.ac.in)

## Links
- Paper![(https://upload.wikimedia.org/wikipedia/commons/e/eb/Paper_icon.svg)]
