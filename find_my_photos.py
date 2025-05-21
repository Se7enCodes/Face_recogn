import cv2
import face_recognition
import os
import shutil
import re
import uuid
import numpy as np
import time
import gradio as gr # <-- Import Gradio
from PIL import Image # <-- To handle image objects from Gradio

# --- Configuration (remains the same) ---
SOURCE_PHOTOS_DIR = "all_photos"
USER_SELFIE_STORAGE_DIR = "user_selfies" # Gradio might save uploaded selfies here too
OUTPUT_BASE_DIR = "sorted_user_photos"
UNKNOWN_FACE_DIR_NAME = "unknown_user"
ENCODINGS_FILE_PATH = "known_faces_encodings.npz"

# Ensure base directories exist (Gradio might run from a different working dir, ensure these are created)
# It's good practice for these paths to be absolute or resolved at runtime if the script's location is variable.
# For simplicity, we assume they are relative to where the script is run.
os.makedirs(SOURCE_PHOTOS_DIR, exist_ok=True)
os.makedirs(USER_SELFIE_STORAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def sanitize_foldername(name):
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[-\s]+', '-', name)
    return name if name else UNKNOWN_FACE_DIR_NAME

def validate_phone_number(phone):
    """Validates phone number: must be 10 digits."""
    if phone.isdigit() and len(phone) == 10:
        return True, ""
    return False, "Invalid phone number. Must be exactly 10 digits."

def validate_email_address(email):
    """Validates email address format."""
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if email and re.match(email_regex, email):
        return True, ""
    return False, "Invalid email format (e.g., user@example.com)."

def load_and_encode_face(image_input, is_path=True):
    """
    Loads an image from a path or uses a PIL Image object,
    finds the first face, and returns its encoding.
    """
    try:
        if is_path:
            # print(f"Loading and encoding face from path: {image_input}") # Console log
            image = face_recognition.load_image_file(image_input)
        else: # image_input is a PIL Image object
            # print(f"Encoding face from PIL Image object.") # Console log
            image = np.array(image_input.convert('RGB')) # Ensure RGB

        face_encodings_list = face_recognition.face_encodings(image)

        if face_encodings_list:
            return face_encodings_list[0]
        else:
            print(f"Warning: No faces found in the provided selfie.")
            return None
    except Exception as e:
        print(f"Error processing selfie image: {e}")
        return None

def generate_and_save_known_encodings(source_dir, encodings_file_path):
    # print(f"Generating and saving new encodings from {source_dir} to {encodings_file_path}...") # Console log
    known_encodings = []
    known_filenames = []
    files_processed = 0
    skipped_files = 0
    start_time = time.time()
    
    # Progress tracking for Gradio (optional, more advanced for real-time updates)
    # For now, we'll just return a summary message.

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(source_dir, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                current_image_encodings = face_recognition.face_encodings(image)
                
                for encoding in current_image_encodings:
                    known_encodings.append(encoding)
                    known_filenames.append(filename) 
                
                # if not current_image_encodings:
                    # print(f"    No faces found in {filename}.") # Console log
                files_processed += 1
            except Exception as e:
                print(f"    Error processing {filename} for encoding: {e}. Skipping.")
                skipped_files +=1
    
    if known_encodings:
        np.savez_compressed(encodings_file_path, encodings=np.array(known_encodings), filenames=np.array(known_filenames))
        end_time = time.time()
        msg = (f"Encoding Generation Complete: Successfully generated {len(known_encodings)} face encodings "
               f"from {files_processed} images ({skipped_files} skipped) in {end_time - start_time:.2f} seconds. "
               f"Encodings saved to {os.path.basename(encodings_file_path)}.")
        print(msg) # Console log
        return known_encodings, known_filenames, msg
    else:
        msg = "Encoding Generation: No encodings were generated. Source directory might be empty, no faces found, or all files had errors."
        print(msg) # Console log
        # Save an empty file to avoid re-scanning empty/problematic dirs constantly if file didn't exist
        np.savez_compressed(encodings_file_path, encodings=np.array([]), filenames=np.array([]))
        return [], [], msg

def load_known_encodings(encodings_file_path):
    try:
        data = np.load(encodings_file_path, allow_pickle=True) 
        if data['encodings'].size == 0 and data['filenames'].size == 0 :
             msg = f"Loaded encoding file '{os.path.basename(encodings_file_path)}' is empty (no faces previously found or empty source dir)."
             # print(msg) # Console log
             return [],[], msg
        encodings = list(data['encodings']) if data['encodings'].size > 0 else []
        filenames = list(data['filenames']) if data['filenames'].size > 0 else []
        msg = f"Successfully loaded {len(encodings)} known face encodings from {os.path.basename(encodings_file_path)}."
        # print(msg) # Console log
        return encodings, filenames, msg
    except FileNotFoundError:
        msg = f"Encodings file '{os.path.basename(encodings_file_path)}' not found. Will proceed to generate if 'Force Rescan' is chosen or if it's the first run."
        # print(msg) # Console log
        return None, None, msg
    except Exception as e:
        msg = f"Error loading encodings from {os.path.basename(encodings_file_path)}: {e}. Will proceed to generate if 'Force Rescan' is chosen."
        print(f"Error loading encodings: {e}") # Console log
        return None, None, msg

def find_matching_photos(selfie_encoding, all_known_encodings, all_known_filenames, source_image_dir):
    if selfie_encoding is None or not all_known_encodings:
        return []
    matched_photo_filenames = set() 
    # print(f"\nComparing selfie with {len(all_known_encodings)} known faces...") # Console log
    start_time = time.time()
    # Ensure all_known_encodings is a list of numpy arrays
    # np.savez_compressed already saves them as numpy arrays, and list() conversion is fine.
    matches_array = face_recognition.compare_faces(all_known_encodings, selfie_encoding, tolerance=0.6)
    for i, match in enumerate(matches_array):
        if match:
            matched_photo_filenames.add(all_known_filenames[i])
    end_time = time.time()
    # print(f"Comparison completed in {end_time - start_time:.2f} seconds.") # Console log
    matched_photo_paths = [os.path.join(source_image_dir, fname) for fname in matched_photo_filenames]
    return matched_photo_paths

def create_user_folder_and_copy_photos(user_identifier, matched_photos, base_output_dir):
    user_folder_name = sanitize_foldername(user_identifier)
    user_specific_dir = os.path.join(base_output_dir, user_folder_name)
    # Ensure the user-specific directory is clean if it already exists, or handle as needed
    if os.path.exists(user_specific_dir):
        shutil.rmtree(user_specific_dir) # Remove old results for this user for this session
    os.makedirs(user_specific_dir, exist_ok=True)
    
    copied_files_paths = []
    copied_count = 0
    if not matched_photos: # Ensure matched_photos is not None
        matched_photos = []

    for photo_path in matched_photos:
        try:
            filename = os.path.basename(photo_path)
            destination_path = os.path.join(user_specific_dir, filename)
            shutil.copy(photo_path, destination_path)
            copied_files_paths.append(destination_path) 
            copied_count += 1
        except Exception as e:
            print(f"  Error copying {os.path.basename(photo_path)}: {e}") # Console log
    
    if copied_count > 0:
        msg = f"Successfully copied {copied_count} matched photos to folder: '{user_folder_name}'"
    else:
        msg = f"No photos were copied to '{user_folder_name}' (no matches found or copy errors)."
    # print(msg) # Console log
    return msg, user_specific_dir, copied_files_paths

# --- Main function for Gradio Interface ---
def process_request_gradio(phone_number, email_address, selfie_pil_image, force_rescan_encodings):
    """
    This function will be called by Gradio.
    It takes user inputs from the UI and returns outputs to be displayed.
    """
    # Initialize outputs for Gradio in case of early return
    final_status_message = "An unexpected error occurred."
    gallery_output = []
    zip_file_output = None
    
    status_messages = ["Process started..."]

    # 1. Validate Inputs
    is_phone_valid, phone_error_msg = validate_phone_number(phone_number)
    if not is_phone_valid:
        return phone_error_msg, None, None 

    is_email_valid, email_error_msg = validate_email_address(email_address)
    if not is_email_valid:
        return email_error_msg, None, None

    if selfie_pil_image is None:
        return "ERROR: Please upload or capture a selfie.", None, None
    
    status_messages.append(f"Inputs Validated: Phone: {phone_number}, Email: {email_address}")

    # 2. Handle Known Encodings (Load or Generate)
    known_encodings, known_filenames, enc_msg = load_known_encodings(ENCODINGS_FILE_PATH)
    status_messages.append(f"Encoding Status: {enc_msg}")

    if force_rescan_encodings or known_encodings is None: # known_encodings is None if file not found or error
        status_messages.append("Force Rescan active or no existing encodings found. Attempting to generate new encodings...")
        if not os.path.exists(SOURCE_PHOTOS_DIR) or not os.listdir(SOURCE_PHOTOS_DIR):
            msg = f"ERROR: Source photo directory '{SOURCE_PHOTOS_DIR}' is empty or does not exist. Cannot generate encodings."
            status_messages.append(msg)
            if known_encodings is None and not os.path.exists(ENCODINGS_FILE_PATH): # If file truly didn't exist, create an empty one
                 np.savez_compressed(ENCODINGS_FILE_PATH, encodings=np.array([]), filenames=np.array([]))
            known_encodings, known_filenames = [], [] # Ensure these are empty lists
            return "\n".join(status_messages), None, None # Critical error, stop here
        else:
            known_encodings, known_filenames, gen_msg = generate_and_save_known_encodings(SOURCE_PHOTOS_DIR, ENCODINGS_FILE_PATH)
            status_messages.append(f"Encoding Generation: {gen_msg}")
    
    if not known_encodings: # Check after attempting load/generate
        # This condition means either source dir was empty, no faces found, or some other error during encoding
        status_messages.append(f"Warning: No face encodings available (source directory '{SOURCE_PHOTOS_DIR}' might be empty, no faces found, or errors occurred). Cannot perform matching.")
        # Don't necessarily exit, as the messages above would indicate the problem.
        # The find_matching_photos will return empty if known_encodings is empty.

    # 3. Process Selfie
    status_messages.append("Processing selfie...")
    selfie_encoding = load_and_encode_face(selfie_pil_image, is_path=False) 

    if selfie_encoding is None:
        status_messages.append("ERROR: Could not process the selfie (no face found or error).")
        return "\n".join(status_messages), None, None
    status_messages.append("Selfie processed successfully.")

    # 4. Find Matching Photos
    if not known_encodings: # If still no encodings (e.g., empty source dir from start)
        status_messages.append("No known face encodings to compare against. No matches possible.")
        matched_photo_paths = []
    else:
        status_messages.append("Searching for your photos using stored encodings...")
        matched_photo_paths = find_matching_photos(selfie_encoding, known_encodings, known_filenames, SOURCE_PHOTOS_DIR)

    if not matched_photo_paths:
        status_messages.append("No matching photos found in the collection for your selfie.")
        # Gallery will be empty, no zip file
        gallery_output = []
        zip_file_output = None
    else:
        status_messages.append(f"Match Found: {len(matched_photo_paths)} photo(s) seem to contain a match.")

        # 5. Create User Folder and Copy Photos
        sanitized_phone = sanitize_foldername(phone_number)
        copy_msg, user_specific_dir, copied_files_local_paths = create_user_folder_and_copy_photos(
            sanitized_phone,
            matched_photo_paths,
            OUTPUT_BASE_DIR
        )
        status_messages.append(f"File Copy: {copy_msg}")
        gallery_output = copied_files_local_paths # Use the paths of copied files for the gallery

        # 6. Create a ZIP file of the matched photos for download
        if copied_files_local_paths:
            try:
                zip_base_name = os.path.join(OUTPUT_BASE_DIR, f"matched_photos_{sanitized_phone}_{uuid.uuid4().hex[:6]}") # Add uuid to avoid name clashes
                # Ensure the directory to zip exists and is not empty
                if os.path.exists(user_specific_dir) and os.listdir(user_specific_dir):
                    shutil.make_archive(zip_base_name, 'zip', user_specific_dir)
                    zip_file_output = f"{zip_base_name}.zip"
                    status_messages.append(f"Download: Matched photos zipped for download.")
                else:
                    status_messages.append("Download: No files were available in the user directory to zip.")
            except Exception as e:
                status_messages.append(f"Download Error: Could not create ZIP file: {e}")
                print(f"Error zipping: {e}") # Console log
        else:
            status_messages.append("Download: No copied files to zip.")
            
    final_status_message = "\n".join(status_messages)
    return final_status_message, gallery_output, zip_file_output


# --- Define Gradio Interface ---
inputs = [
    gr.Textbox(label="Your 10-digit Phone Number", placeholder="e.g., 1234567890", info="Used for organizing your photos."),
    gr.Textbox(label="Your Email Address", placeholder="e.g., user@example.com", info="Please provide your email."),
    gr.Image(type="pil", label="Upload or Capture Your Selfie", sources=["upload", "webcam"], height=400),
    gr.Checkbox(label="Force Rescan of Source Photo Encodings", info="Check this if new event photos have been added or if this is the very first run. This process can be slow depending on the number of source photos.")
]

outputs = [
    gr.Textbox(label="Process Status & Messages", lines=10), # Increased lines for more messages
    gr.Gallery(label="Matched Photos Preview", height=600, object_fit="contain", columns=5, preview=True),
    gr.File(label="Download Matched Photos (ZIP file)")
]

description = (
    "## Welcome to the Find My Photos Service!\n\n"
    "1.  Enter your **10-digit phone number** and **email address**.\n"
    "2.  **Upload a clear selfie** or use your webcam to capture one. You can crop/edit it after upload.\n"
    "3.  Click **Submit**.\n\n"
    "The system will then find photos of you from the event collection.\n\n"
    "**Important Note for Admins/First Use:** If new event photos have been added to the system's `all_photos` folder, "
    "or if this is the first time running the service with a new set of event photos, "
    "please check the **'Force Rescan of Source Photo Encodings'** box. This initial scan and encoding process "
    "can take several minutes depending on the number of photos. Subsequent runs (without 'Force Rescan') will be much faster."
)

article = (
    "<div style='text-align: center; margin-top: 20px;'>"
    "<p>Powered by Python, Gradio, and Face Recognition technology.</p>"
    "<p><i>Please ensure your selfie is clear and well-lit for best results.</i></p>"
    "</div>"
)

iface = gr.Interface(
    fn=process_request_gradio,
    inputs=inputs,
    outputs=outputs,
    title="📸 Find My Photos!",
    description=description,
    article=article,
    allow_flagging='never',
    theme=gr.themes.Soft() # Using a built-in theme
)

# --- To run the Gradio app ---
if __name__ == "__main__":
    # Print some helpful paths for the user running the script
    print("--- Application Paths ---")
    print(f"Source photos are expected in: '{os.path.abspath(SOURCE_PHOTOS_DIR)}'")
    print(f"User selfies (if saved from upload, though Gradio handles temp files) would be in: '{os.path.abspath(USER_SELFIE_STORAGE_DIR)}'")
    print(f"Sorted photos and ZIP files will be saved in subdirectories of: '{os.path.abspath(OUTPUT_BASE_DIR)}'")
    print(f"Pre-computed encodings file: '{os.path.abspath(ENCODINGS_FILE_PATH)}'")
    print("-------------------------")
    print("\nLaunching Gradio app... Access it locally via the URL printed below (usually http://127.0.0.1:7860 or similar).")
    
    # To make it accessible on your local network: iface.launch(server_name="0.0.0.0")
    # To create a temporary public link (expires in 72 hours): iface.launch(share=True)
    iface.launch(share=True)
