import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import ttk
import os
from PIL import Image, ImageTk

# Function to move focus to the next entry when the Enter key is pressed
def focus_next_entry(event):
    # This function is used to move focus to the next entry when the Enter key is pressed
    event.widget.tk_focusNext().focus()

# Function to format the confusion matrix for display
def format_confusion_matrix(matrix):
    # Add handling for missing classes in the confusion matrix
    labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    matrix = matrix.ravel()
    formatted_matrix = [f'{label}: {value}' for label, value in zip(labels, matrix)]
    return '\n'.join(formatted_matrix)

# Function to format the classification report for display
def format_classification_report(report):
    # Add handling for missing classes in the classification report
    lines = report.split('\n')
    header = lines[0]
    lines = lines[2:-3]
    formatted_report = [header] + lines
    return '\n'.join(formatted_report)

# Function to display the prediction results in a new window
def show_results_window(conf_matrix, cls_report, sample_prediction):
    # Create a new window to display the results
    results_window = tk.Toplevel(root)
    results_window.title("Prediction Results")
    
    # Set the logo for the main window
    logo_path = os.path.join(os.getcwd(), "logo.ico")
    results_window.iconbitmap(logo_path)

    # Calculate the center position of the results window
    window_width = 600  # Adjust this value as needed
    window_height = 400  # Adjust this value as needed
    screen_width = results_window.winfo_screenwidth()
    screen_height = results_window.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # Set the window size and position
    results_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Create a Text widget to display the results
    text_widget = tk.Text(results_window, wrap=tk.WORD, padx=20, pady=10)
    text_widget.pack(fill=tk.BOTH, expand=True)

    # Format and display the confusion matrix
    formatted_conf_matrix = format_confusion_matrix(conf_matrix)
    text_widget.insert(tk.END, "Confusion Matrix:\n")
    text_widget.insert(tk.END, formatted_conf_matrix)

    # Format and display the classification report
    formatted_cls_report = format_classification_report(cls_report)
    text_widget.insert(tk.END, "\n\nClassification Report:\n")
    text_widget.insert(tk.END, formatted_cls_report)

    # Display the prediction result
    text_widget.insert(tk.END, f"\n\nPrediction Result: {sample_prediction[0]}")
    
    # Determine the risk of diabetes based on the prediction
    if sample_prediction[0] == 0:
        text_widget.insert(tk.END, "\n\nBased on the prediction, it appears that you are not at risk of diabetes.")
    elif sample_prediction[0] == 1:
        text_widget.insert(tk.END, "\n\nBased on the prediction, it appears that you may be at high risk of diabetes. We recommend consulting a doctor for further evaluation.")

# Function to train the model, make predictions, and display the results
def train_and_predict_model():
    try:
        # Load the diabetes dataset into a DataFrame named "df"
        df = pd.read_csv('diabetes-dataset.csv')

        # Replace empty entries in entry fields with 0
        sample_data = [float(entry.get()) if entry.get() else 0 for entry in entry_fields]

        # Separate input features (X) and target variable (y)
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=10, shuffle=True)

        # Initialize the RandomForestClassifier with adjusted hyperparameters
        model = RandomForestClassifier(
            n_estimators=500,  # Number of trees in the forest
            max_depth=10,      # Maximum depth of the tree
            min_samples_split=2,
            min_samples_leaf=1,
        )

        # Train the RandomForestClassifier on the training data
        model.fit(X_train, y_train)

        # Predict the target variable for the test set
        y_pred = model.predict(X_test)

        # Generate the confusion matrix to evaluate model performance
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Generate the classification report to get more detailed performance metrics
        cls_report = classification_report(y_test, y_pred)

        # Make a prediction for the sample data point
        sample_prediction = model.predict([sample_data])

        # Open a new window to show the results
        show_results_window(conf_matrix, cls_report, sample_prediction)

        # Remove any existing warning
        warning_label.config(text="")
    except ValueError as e:
        # Display a warning when non-numeric inputs are entered
        warning_label.config(text="Invalid input! Please enter numeric values only.")

# Function to reset the entry fields
def reset_entries():
    # Clear the entry fields
    for entry in entry_fields:
        entry.delete(0, tk.END)

def fade_out(window, alpha):
    if alpha > 0:
        alpha -= 0.05
        window.attributes('-alpha', alpha)
        window.after(50, fade_out, window, alpha)
    else:
        window.withdraw()

def fade_in(window, alpha):
    if alpha < 1:
        alpha += 0.05
        window.attributes('-alpha', alpha)
        window.after(50, fade_in, window, alpha)

def go_to_main_code():
    global instruction_window
    fade_out(instruction_window, 1.0)  # Start fading out the instruction window
    instruction_window.after(800, start_main_code)  # Start the main code after 800ms delay

def start_main_code():
    global root, instruction_window
    root = tk.Tk()
    root.title("SmartDiaSense")
    root.geometry("400x500")  # Set the window size
    root.resizable(width=False, height=False)
    
    # Set the logo for the main window
    logo_path = os.path.join(os.getcwd(), "logo.ico")
    root.iconbitmap(logo_path)

    # Create the frame for the labels and entry fields
    input_frame = ttk.Frame(root, padding="20")
    input_frame.pack(pady=10)

    # Create the labels and entry fields
    labels = [
        "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
        "Insulin", "BMI", "Diabetes Pedigree Function", "Age"
    ]

    global entry_fields
    entry_fields = []  # Make entry_fields a global variable

    for i, label_text in enumerate(labels):
        label = ttk.Label(input_frame, text=label_text, anchor="center")
        label.grid(row=i, column=0, padx=5, pady=5, sticky="ew")

        entry = ttk.Entry(input_frame)
        entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
        entry_fields.append(entry)
        entry.bind("<Return>", focus_next_entry)  # Bind the Enter key to switch focus to the next entry

    # Create the Predict Diabetes button
    predict_button = ttk.Button(root, text="Predict Diabetes", command=train_and_predict_model)
    predict_button.pack(pady=10, padx=20, fill="x")

    # Create the Reset button
    reset_button = ttk.Button(root, text="Reset", command=reset_entries)
    reset_button.pack(pady=10, padx=20, fill="x")

    # Create a frame for the results and warnings
    results_frame = ttk.Frame(root)
    results_frame.pack(pady=10)

    # Create a label to display warnings
    global warning_label
    warning_label = ttk.Label(root, text="", foreground="red")
    warning_label.pack(pady=5)

    # Center the window on the screen
    root.eval('tk::PlaceWindow . center')

    # Start the Tkinter event loop
    root.mainloop()
    
    # Destroy the instruction window after fading it out completely
    instruction_window.destroy()

    # Fade in the main window
    root.attributes('-alpha', 0.0)
    root.deiconify()
    fade_in(root, 0.0)

# Create the instruction window
instruction_window = tk.Tk()
instruction_window.title("Instructions")
instruction_window.geometry("400x500")  # Set the window size
instruction_window.resizable(width=False, height=False)

# Set the logo for the main window
logo_path = os.path.join(os.getcwd(), "logo.ico")
instruction_window.iconbitmap(logo_path)

# Explanation of Features
feature_explanation = """Instructions and Notes:
Please enter numeric values for each feature:

1. Pregnancies: The number of times the patient has been pregnant.
2. Glucose: The plasma glucose concentration in a 2-hour oral glucose tolerance test.
3. Blood Pressure: The diastolic blood pressure (mm Hg).
4. Skin Thickness: The skinfold thickness of the triceps (mm).
5. Insulin: The 2-hour serum insulin level (mu U/ml).
6. BMI (Body Mass Index): The weight in kilograms divided by the square of the height in meters.
7. Diabetes Pedigree Function: A function that represents the likelihood of diabetes based on family history.
8. Age: The age of the patient in years.

Leaving a field empty will assume a value of 0.

Press the 'Predict Diabetes' button to make a prediction.

The results will be displayed in a new window."""

# Create a label to display the feature explanations
feature_explanation_label = ttk.Label(instruction_window, text=feature_explanation, wraplength=350, justify="left")
feature_explanation_label.pack(pady=20, padx=10)

# Create the Start button to go to the main code
start_button = ttk.Button(instruction_window, text="Start", command=go_to_main_code)
start_button.pack(pady=10, padx=20, fill="x")

# Center the instruction window on the screen
instruction_window.eval('tk::PlaceWindow . center')

# Start the Tkinter event loop for the instruction window
instruction_window.mainloop()
