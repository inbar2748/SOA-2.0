# slim SOA App Installation Guide

This guide will help you set up Python and install all required packages for the image processing application. Don't worry if you're new to programming - we'll go through this step by step!

## Step 1: Installing Python 3.9 on Windows

1. Download Python 3.9:
   - Go to [Python Downloads Page](https://www.python.org/downloads/windows/)
   - Scroll down to find Python 3.9 (Any 3.9.x version will work)
   - Click on "Windows installer (64-bit)"

2. Install Python:
   - Open the downloaded file (it should be named something like `python-3.9.x-amd64.exe`)
   - ✅ **IMPORTANT**: Check the box that says "Add Python 3.9 to PATH"
   - Click "Install Now"
   - Wait for the installation to complete
   - Click "Close" when finished

3. Verify the installation:
   - Open Command Prompt (you can search for "cmd" in the Start menu)
   - Type: `python --version`
   - You should see something like `Python 3.9.x`

## Step 2: Setting Up Your Project

1. Create a project folder:
   - Create a new folder on your computer (name it whatever you like)
   - Open Command Prompt
   - Navigate to your folder using the `cd` command. For example:
     ```
     cd C:\Users\YourName\Documents\MyProject
     ```

2. Create a virtual environment:
   - In the Command Prompt, type:
     ```
     python -m venv venv
     ```
   - This creates a new folder called `venv` in your project folder

3. Activate the virtual environment:
   - In the Command Prompt, type:
     ```
     venv\Scripts\activate
     ```
   - You should see `(venv)` appear at the beginning of the command line

## Step 3: Installing Required Packages

1. Create requirements.txt:
   - Create a new file in your project folder called `requirements.txt`
   - Copy and paste these lines into it:
     ```
     opencv-python>=4.6.0,<4.7.0
     numpy>=1.24.3,<1.25.0
     Pillow>=9.4.0,<9.5.0
     scikit-image>=0.19.3,<0.20.0
     ```

2. Install the packages:
   - Make sure your virtual environment is activated (you should see `(venv)` at the start of the command line)
   - Type:
     ```
     pip install -r requirements.txt
     ```
   - Wait for all packages to install (this might take a few minutes)

## Troubleshooting

If you get any errors:
- Make sure Python is added to PATH (you might need to reinstall Python and check the box)
- Make sure you're in the correct folder in Command Prompt
- Make sure the virtual environment is activated (you should see `(venv)`)
- Try running Command Prompt as Administrator

## Need Help?

If you're having problems:
1. Read any error messages carefully
2. Make sure you followed each step exactly
3. Ask your teacher or instructor for help

## Starting Fresh

If you need to start over:
1. Delete the `venv` folder
2. Follow the steps from "Step 2: Setting Up Your Project" again
