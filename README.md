CSV Folder:
Contains CSV Files
Classification - survey lung cancer
Regression - weather

Models Folder:
Contains 4 Models (2 Models for each Classification and Regression)

Pages Folder:
Contains separate py files for the activity:
1 - Classification Training Models
2 - Classification Performance Metrics
3 - Linear Regression Training Models
4 - Linear Regression Performance Metrics
Dashboard - Overview of the CSV Files

Features:
Dashboard
* Displays Insightful data for the CSV Files
* Displays a summary of graphs for each CSV File
* Overview and a sample of what the CSV Files contain

1 - Classification Resampling Techniques
* Trains Models using the survey lung cancer CSV file, the user can choose to train using K-Fold or Leave One Out Cross Validation
* User can choose / pick K-Fold number of Folds, Leave One Out Cross Validation is default one instance
* Lung Cancer Diagnosis Predictor - Predicts if a person has lung cancer based on the questions / symptoms
* Target Column = "LUNG_CANCER" (What the model Predicts)
* Feature Column = "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC_DISEASE",
        "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING",
        "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN" (What the prediction will base off)
  
2 - Classification Performance Metrics
*Classification Accuracy: Measures the percentage of correct predictions made by a model.
*Log Loss: Evaluates the model's prediction confidence by penalizing incorrect predictions with probabilities closer to zero.
*Confusion Matrix: A table that shows the true positives, true negatives, false positives, and false negatives of a model.
*Classification Report: Summarizes precision, recall, F1-score, and support for each class in a classification model.
*ROC Curve: A graph showing the trade-off between the true positive rate (sensitivity) and false positive rate at various thresholds.

3. Linear Regression Resampling Techniques
* Trains Models using the weather CSV file, the user can choose to train using Split into Train and Test Sets or Repeated Random Train-Test Splits
* User can choose / pick Test / Split Sizes depending on what sampling technique is used
* Temperature Predictor - Predicts the temperature based on the questions
* Target Column = "Temperature_c" (What the model Predicts)
* Feature Column = "Humidty" "Wind_Speed_kmh" "Wind_Bearing_degrees" "Visibility_km" "Pressure_millibars" "Rain" "Description" (What the prediction will base off)

4. Linear Regression Performance Metrics
* MSE (Mean Squared Error): Measures the average squared difference between the predicted values and actual values, emphasizing larger errors due to squaring.
* MAE (Mean Absolute Error): Measures the average absolute difference between predicted values and actual values, providing a straightforward interpretation of error magnitude.
* R² (Coefficient of Determination): Indicates how well the model explains the variance in the target variable, with values closer to 1 showing better fit.

How to use:
2 Methods
1 - Clone Method 
* create a Fresh New Folder on your Desktop
* open that folder (make sure the directory is of that folder) on VS Code
* type on the terminal of the VS Code "git clone ⁦https://github.com/jeahpearl/Lab2ITD105⁩"
* If all goes well, you should have a clone of this repository

2 - Manual Method
* create a Fresh New Folder on your Desktop
* go to this repository "https://github.com/jeahpearl/Lab2ITD105"
* click on the green button that says "<> Code"
* wait for the zip to download, extract the zip on your Fresh New Folder

After having the files on the folder, just type on the terminal
"streamlit run Dashboard.py"

If you have complete packages, then you should have no errors and the streamlit app will run smoothly
P.S don't forget to change the save model, csv file paths on the py files
