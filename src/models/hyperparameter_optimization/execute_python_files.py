import os
import subprocess

def run_python_files(paths):
    try:
        # Get the name of this execution file
        execution_file = os.path.abspath(__file__)  # Use absolute path for comparison

        for path in paths:
            if os.path.isdir(path):  # If the path is a folder
                print(f"Processing folder: {path}")

                python_files = [f for f in os.listdir(path) if f.endswith('.py')]
                python_files.sort()

                for python_file in python_files:
                    # Skip the execution file
                    file_path = os.path.abspath(os.path.join(path, python_file))
                    if file_path == execution_file:
                        print(f"Skipping execution file: {python_file}")
                        continue

                    run_python_file(file_path)

            elif os.path.isfile(path) and path.endswith('.py'):  # If the path is a file
                file_path = os.path.abspath(path)  # Normalize path
                print(f"Processing file: {file_path}")
                run_python_file(file_path)
            else:
                print(f"Invalid path (not a folder or Python file): {path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def run_python_file(file_path):
    try:
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            return

        print(f"Running: {file_path}")
        working_dir = os.path.dirname(file_path)

        # Run the Python file
        result = subprocess.run(
            ["python", file_path],
            cwd=working_dir,  # Set working directory
            capture_output=True,
            text=True
        )

        print("Output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)

    except Exception as e:
        print(f"An error occurred while running {file_path}: {e}")

if __name__ == "__main__":
    paths = [
        'GRU-LSTM_classification_bo.py',
        'GRU-LSTM_classification_rl.py',
        'GRU-LSTM_regression_bo.py',
        'TCN-LSTM_classification_bo.py',
        'TCN-LSTM_classification_rl.py',
        'TCN-LSTM_regression_bo.py'
    ]
    run_python_files(paths)
