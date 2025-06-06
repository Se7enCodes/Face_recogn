# AI-Powered Photo Finder & Organizer:

## Project Description

This project is a Python application designed to help users quickly find all their photos from a collection of event images using a selfie. It leverages face recognition technology to identify and group photos of a specific individual. The application provides a user-friendly web interface built with Gradio, allowing users to input their details, provide a selfie via webcam, specify event photo folders, and then view and download their matched photos.

The system is optimized for performance by pre-calculating and storing face encodings from the event photos, significantly speeding up the matching process for subsequent uses.

## How It Works

User Input: The user provides their phone number, email address, and specifies which event subfolders (within a master event directory) they want to search.

Selfie Capture: The user captures a selfie using their webcam through the Gradio interface.

Face Encoding (Selfie): The system processes the captured selfie to generate a unique numerical representation (face encoding) of the user's face.

Event Photo Encodings (Pre-computation & Loading):

If processing the selected event subfolders for the first time (or if a "Force Rescan" is triggered), the application scans all images within these subfolders.

It detects faces in each event photo and calculates their face encodings.

These encodings, along with their corresponding relative filenames (e.g., subfolder_name/image.jpg), are saved to a local file (known_faces_encodings.npz).

On subsequent runs (for the same set of subfolders and without "Force Rescan"), these pre-computed encodings are loaded directly from the file, saving significant processing time.

Face Matching: The selfie's face encoding is compared against all the loaded/generated face encodings from the selected event subfolders. A tolerance level is used to determine if two faces are a match.

## Output & Download:

Matched photos are copied to a new, user-specific folder (named using their phone number) within an output directory.

The Gradio interface displays a gallery preview of the matched photos.

A ZIP file containing all matched photos is created and made available for download.

## Running the Application Locally

Follow these steps to set up and run the application on your local machine.

#### 1. Create a Conda Environment
It's recommended to use a Conda environment to manage dependencies.

Create a new Conda environment (e.g., named "photo_finder_env") with Python 3.9
conda create -n photo_finder_env python=3.9 -y

Activate the environment
conda activate photo_finder_env

#### 2. Install Dependencies
With the Conda environment activated, install the necessary Python libraries. The core dependencies are dlib, face_recognition, opencv-python, numpy, Pillow, and gradio.

It's often best to install dlib via conda-forge first
conda install -c conda-forge dlib -y

Then install the rest using pip
pip install face_recognition opencv-python Pillow numpy gradio

(Note: If pip install face_recognition has issues with dlib compilation, ensure cmake is also installed in your system or environment: conda install -c conda-forge cmake -y before installing dlib and face_recognition)

#### 3. Set Up Folder Structure
Create the following directory structure in your project folder (where you save the Python script, e.g., app.py):

your_project_folder/

├── app.py                         # The main Python script with Gradio interface

├── Event_Photos_Root/             # MASTER DIRECTORY for all event photos

│   ├── event_A_photos/            # Example subfolder for Event A

│   │   ├── image1.jpg

│   │   └── image2.png

│   ├── brides_folder/             # Example subfolder for Bride's photos

│   │   ├── image3.jpg

│   │   └── ...

│   └── ...                        # Other event subfolders

└── (Other files will be generated by the script):

    ├── known_faces_encodings.npz  # Stores pre-computed face encodings
    
    ├── user_selfies/              # Stores temporary selfies (if saved)
    
    └── sorted_user_photos/        # Output directory for user-specific matched photos
    
        └── <phone_number>/        # User-specific folder with matched photos
        
            ├── matched_photo1.jpg
            
            └── ...

Create the Event_Photos_Root/ directory.

Inside Event_Photos_Root/, create subfolders for different events or categories (e.g., event_A_photos, brides_folder) and place the respective event images into these subfolders.

#### 4. Running the Script
Navigate to your project folder in your terminal (with the Conda environment activated) and run the Python script:

Ensure your Conda environment is active:
  conda activate photo_finder_env

Navigate to the directory containing app.py and Event_Photos_Root/
  cd path/to/your_project_folder

Run the script
  python app.py

#### 5. Accessing the Gradio Interface
After running the script, the terminal will display messages including a local URL, typically:

Running on local URL:  [http://127.0.0.1:7860](http://127.0.0.1:7860)

Open this URL in your web browser.

You will see the "Find My Photos!" Gradio interface.

Follow the instructions on the interface:

Enter your phone number and email.

Enter the names of the event subfolders (located inside Event_Photos_Root/) you want to search, separated by commas (e.g., event_A_photos, brides_folder).

Capture your selfie using the webcam.

Important: If it's your first time processing these specific subfolders, or if the photos within them have changed, check the "Force Rescan..." box. This initial scan can take time.

Click "Submit".

The application will then process your request, and you'll see status messages, a preview gallery of matched photos, and a download link for a ZIP file of your photos.

This project uses the face_recognition library by Adam Geitgey, which is built using dlib's state-of-the-art face recognition built with deep learning.
