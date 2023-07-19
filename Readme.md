# Setup for git 
- git lfs install
- git lfs track "*.pkl"

# Setup
- install python 3.10.6
- git clone https://github.com/haovu429/recommend.git
- At recommend directory, run "python -m venv venv" command

- At recommend directory,
    + With window, run "./venv/Scripts/activate" command to active environment
    + With ubuntu, run "source venv/bin/activate" command to active environment
- At recommend directory, run "pip install -r requirements.txt" command to active environment
- At recommend directory, run "python -u predict.py" command to start project, Flask app run at port 5000