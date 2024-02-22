# 2024_imat
## IMAT - Interactive Melody Annotation Tool
Tool for correcting melody from polyphonic audio

How to use the tool?

**Libraries**
- Download the entire folder on your local system.
- Create a new environment and run the requirement.txt file to install the required libraries in the new environment.

**Run**
- On the terminal, activate the new environment and run 'app.py'.
- After executing 'app.py', copy the link that appears on the terminal and open it on the browser, or press Ctrl+link to open it directly on the browser.
  
- **STEP 1:** Click on 'Upload audio', and browse through your system to annotate the audio you wish. Please make sure you upload a '.wav' audio file.
- **STEP 2:** Once the audio is uploaded, click on the 'Show Spectrogram' button to visualize the spectrogram.
- **STEP 3:** To visualize the estimated melody, click on the 'Show Melody' checkbox. The spectrogram overlayed with the estimated melody appears.
- **STEP 4:** The user can correct the melody as follows:
1. The user can select a particular range of the spectrogram to annotate the extracted melody. Once selected, click on 'Annotate'.
2. After clicking on 'Annotate', another interactive plot appears, which contains the estimated melody (at 10ms hop size) in that range. The melody at each time frame is an anchor point that can modified as per user. Another bar plot appears just below the interactive plot that contains confidence values at each time frame in that range.
3. The user can visually observe the low-confidence frames and manually correct the melody in those frames. The user can either drag a single melody anchor point or select consecutive anchor points (by clicking and dragging a rectangle over the anchor points).
4. If the user selects multiple consecutive anchor points and clicks on 'Remove Pitches', the entire anchor points will map to zero, indicating an absence of a singing voice. **Don't forget to right-click on the interactive plot after selecting multiple anchor points**(it deselects the selected range of anchor points). The corrections can be validated through auditory feedback.
5. Similarly, the user may correct the melody corresponding to the low confidence frames in the entire audio and validate through auditory feedback.
6. After this, the user clicks on the 'Retrain Model' button to adapt the model to these corrections.
7. The retrained model predicts an updated estimated melody, and the user can visualize it as in STEP 3. Also, the confidence values at each time frame are also updated.
8. The points from 1 to 7 are repeated until the user can visually observe that very few melody estimations (less than ~2%) are wrong.
9. To improve the melody precision further, the user always has the option of correcting the remaining melody manually.
- **STEP 5:** Once the user is satisfied with the corrections, the annotations can be downloaded by clicking on the 'Download CSV' button.

**Follow the same procedure to annotate any polyphonic audio.**
