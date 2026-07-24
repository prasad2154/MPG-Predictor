# MPG-Predictor
Vehicle MPG Prediction App
An interactive Streamlit machine learning application that predicts vehicle fuel efficiency in Miles Per Gallon (MPG) using a pre-trained model stored in model.pkl.

The application accepts four vehicle features:

Number of cylinders (cyl)
Engine displacement (disp)
Vehicle weight (wt)
Horsepower (hp)
Features
Interactive Streamlit user interface
MPG prediction using a pre-trained model
Vehicle preset options
MPG gauge visualization
Fuel-efficiency category
MPG to litres per 100 km conversion
Prediction history
CSV download option
Model loading and input validation
Responsive dashboard layout
Project Structure
mpg_streamlit_app/
├── app.py
├── model.pkl
├── requirements.txt
└── README.md
File Description
File	Description
app.py	Main Streamlit application
model.pkl	Pre-trained MPG prediction model
requirements.txt	Required Python packages
README.md	Project documentation
Input Features
Feature	Description
cyl	Number of engine cylinders
disp	Engine displacement
wt	Vehicle weight
hp	Engine horsepower
The application sends the features to the model in this exact order:

["cyl", "disp", "wt", "hp"]
The output is:

mpg
where mpg means Miles Per Gallon.

Requirements
The following packages are required:

streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.20.0
Installation on Windows using VS Code
1. Open the project folder
Open Visual Studio Code and select:

File → Open Folder
Choose the folder containing:

app.py
model.pkl
requirements.txt
README.md
2. Open the VS Code terminal
Use:

Terminal → New Terminal
3. Create a virtual environment
python -m venv .venv
4. Activate the virtual environment
.\.venv\Scripts\Activate.ps1
If PowerShell blocks the activation command, run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Then activate the environment again:

.\.venv\Scripts\Activate.ps1
5. Select the Python interpreter
Press:

Ctrl + Shift + P
Search for:

Python: Select Interpreter
Select:

.venv\Scripts\python.exe
6. Install the required packages
python -m pip install -r requirements.txt
Run the Application
Use the following command in the VS Code terminal:

python -m streamlit run app.py
The application should open automatically in your browser.

Default local address:

http://localhost:8501
Model Requirements
The model.pkl file must:

Be placed in the same folder as app.py
Accept exactly four input features
Use the feature order shown below
["cyl", "disp", "wt", "hp"]
The model should return a numerical MPG prediction.

Example Input
Feature	Example Value
Cylinders	4
Displacement	140.8
Weight	3.150
Horsepower	95
The predicted MPG depends on the trained model stored in model.pkl.

Common Errors
Streamlit command is not recognized
Error:

streamlit is not recognized as the name of a cmdlet
Install Streamlit:

python -m pip install streamlit
Run the application using:

python -m streamlit run app.py
model.pkl not found
Error:

model.pkl was not found
Make sure the folder contains:

app.py
model.pkl
Both files must be in the same directory.

Feature count mismatch
Error:

X has 4 features, but LinearRegression is expecting 3 features
This means the saved model.pkl was trained using a different number of features.

The model used with this application must expect:

["cyl", "disp", "wt", "hp"]
Use a compatible model.pkl that accepts all four features.

Missing Python package
Install all dependencies again:

python -m pip install -r requirements.txt
Important Notes
The input units must be the same as those used when the model was trained.
Do not change the feature order unless the saved model expects a different order.
Keep model.pkl in the same folder as app.py.
The model is loaded directly from the pickle file.
This application is intended for educational and demonstration purposes.
Deployment
The application can be deployed using:

Streamlit Community Cloud
Render
Railway
AWS
Azure
Google Cloud Platform
For deployment, upload these files:

app.py
model.pkl
requirements.txt
README.md
Author
Yameen Hakim
