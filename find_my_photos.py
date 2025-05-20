

#-------------------------------------------------------------------------------------------------------------------

''' Welcome To Photo Sorter!!! '''

import cv2
import face_recognition
import os
import shutil
import re # For sanitizing phone number for folder name
import uuid # For unique selfie filenames

# --- Configuration ---
SOURCE_PHOTOS_DIR = "all_photos"  # Directory with all photos to search
USER_SELFIE_STORAGE_DIR = "user_selfies" # Optional: where to save user selfies
OUTPUT_BASE_DIR = "sorted_user_photos" # Base directory for storing sorted photos
UNKNOWN_FACE_DIR_NAME = "unknown_user" # Fallback directory name

# Ensure base directories exist
os.makedirs(SOURCE_PHOTOS_DIR, exist_ok=True)
os.makedirs(USER_SELFIE_STORAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def sanitize_foldername(name):
    """Removes or replaces characters not suitable for folder names."""
    name = re.sub(r'[^\w\s-]', '', name).strip() # Keep alphanumeric, spaces, hyphens
    name = re.sub(r'[-\s]+', '-', name) # Replace spaces/multiple hyphens with single hyphen
    return name if name else UNKNOWN_FACE_DIR_NAME

def get_phone_number():
    """Prompts the user for their phone number."""
    while True:
        phone = input("Please enter your phone number: ").strip()
        if phone: # Basic check, can be improved with regex for phone format
            return phone
        else:
            print("Phone number cannot be empty. Please try again.")

def capture_selfie_from_webcam(phone_number_sanitized):
    """Captures a selfie from the webcam."""
    cap = cv2.VideoCapture(0) # 0 is usually the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("\nWebcam activated. Press 's' to save the selfie, or 'q' to quit.")
    selfie_path = None
    unique_id = uuid.uuid4().hex[:8] # short unique id
    filename = f"selfie_{phone_number_sanitized}_{unique_id}.jpg"
    selfie_save_path = os.path.join(USER_SELFIE_STORAGE_DIR, filename)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('Press "s" to save selfie, "q" to quit', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            try:
                cv2.imwrite(selfie_save_path, frame)
                print(f"Selfie saved as {selfie_save_path}")
                selfie_path = selfie_save_path
                break
            except Exception as e:
                print(f"Error saving selfie: {e}")
                selfie_path = None # Ensure it's None if save fails
                break
        elif key == ord('q'):
            print("Selfie capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return selfie_path


def get_selfie():
    """Asks user to upload a selfie or capture one."""
    while True:
        choice = input("How would you like to provide your selfie?\n"
                       "1. Upload an existing photo file\n"
                       "2. Capture a new one from webcam\n"
                       "Enter choice (1 or 2): ").strip()
        if choice == '1':
            selfie_path = input("Enter the full path to your selfie image: ").strip()
            if os.path.isfile(selfie_path):
                # Optionally, copy the uploaded selfie to our storage
                try:
                    filename = f"uploaded_selfie_{uuid.uuid4().hex[:8]}{os.path.splitext(selfie_path)[1]}"
                    stored_selfie_path = os.path.join(USER_SELFIE_STORAGE_DIR, filename)
                    shutil.copy(selfie_path, stored_selfie_path)
                    print(f"Selfie copied to {stored_selfie_path}")
                    return stored_selfie_path # Return the path of the copied selfie
                except Exception as e:
                    print(f"Error copying selfie: {e}. Using original path.")
                    return selfie_path # Fallback to original path if copy fails
            else:
                print("Invalid file path. Please try again.")
        elif choice == '2':
            # We need a temporary identifier for the selfie name if captured before phone number
            # Or we can ask for phone number first. Let's assume phone number is asked first.
            # For this example, phone_number will be passed if needed for filename
            # but capture_selfie_from_webcam handles its own naming now.
            print("Preparing webcam...")
            return capture_selfie_from_webcam("temp_user") # "temp_user" or pass actual phone
        else:
            print("Invalid choice. Please enter 1 or 2.")


def load_and_encode_face(image_path):
    """Loads an image, finds the first face, and returns its encoding."""
    try:
        print(f"Loading and encoding face from: {image_path}")
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            return face_encodings[0]  # Use the first face found
        else:
            print(f"Warning: No faces found in {image_path}")
            return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def find_matching_photos(selfie_encoding, source_dir):
    """
    Scans photos in source_dir, compares them with selfie_encoding,
    and returns a list of paths to matching photos.
    """
    if selfie_encoding is None:
        return []

    matched_photo_paths = []
    print(f"\nScanning photos in {source_dir}...")

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(source_dir, filename)
            print(f"  Checking: {filename}")

            # It's more efficient to load image and get encodings once
            try:
                unknown_image = face_recognition.load_image_file(image_path)
                unknown_encodings = face_recognition.face_encodings(unknown_image)

                if not unknown_encodings:
                    # print(f"    No faces found in {filename}. Skipping.")
                    continue

                # Compare each face found in the unknown image with the selfie encoding
                for unknown_encoding in unknown_encodings:
                    match = face_recognition.compare_faces([selfie_encoding], unknown_encoding, tolerance=0.6) # Adjust tolerance as needed
                    if match[0]: # compare_faces returns a list of booleans
                        print(f"    MATCH FOUND: {filename}")
                        matched_photo_paths.append(image_path)
                        break # Move to next image once a match is found in this one
            except FileNotFoundError:
                print(f"    Error: File not found {image_path}. Skipping.")
            except Exception as e:
                print(f"    Error processing {filename}: {e}. Skipping.")
    return matched_photo_paths

def create_user_folder_and_copy_photos(user_identifier, matched_photos, base_output_dir):
    """Creates a user-specific folder and copies matched photos into it."""
    user_folder_name = sanitize_foldername(user_identifier)
    user_specific_dir = os.path.join(base_output_dir, user_folder_name)

    os.makedirs(user_specific_dir, exist_ok=True)
    print(f"\nCreating folder for you at: {user_specific_dir}")

    copied_count = 0
    for photo_path in matched_photos:
        try:
            shutil.copy(photo_path, user_specific_dir)
            copied_count += 1
            # print(f"  Copied: {os.path.basename(photo_path)}")
        except Exception as e:
            print(f"  Error copying {os.path.basename(photo_path)}: {e}")

    if copied_count > 0:
        print(f"\nSuccessfully copied {copied_count} matched photos to {user_specific_dir}")
    else:
        print(f"\nNo photos were copied to {user_specific_dir} (either no matches or copy errors).")
    return user_specific_dir

def main():
    """Main function to orchestrate the process."""
    print("Welcome to the Photo Finder Service!")
    print("------------------------------------")

    # 1. Get phone number
    phone_number = get_phone_number()
    sanitized_phone_for_folder = sanitize_foldername(phone_number) # For folder naming

    # 2. Get user selfie
    print("\n--- Selfie Time! ---")
    selfie_path = None
    while selfie_path is None: # Loop until a valid selfie is provided or captured
        selfie_input_mode = input("How would you like to provide your selfie?\n"
                                  "1. Upload an existing photo file\n"
                                  "2. Capture a new one from webcam\n"
                                  "Enter choice (1 or 2): ").strip()
        if selfie_input_mode == '1':
            temp_path = input("Enter the full path to your selfie image: ").strip()
            if os.path.isfile(temp_path):
                # Store the uploaded selfie with a unique name
                try:
                    ext = os.path.splitext(temp_path)[1]
                    if not ext: ext = ".jpg" # default extension if none
                    filename = f"uploaded_selfie_{sanitized_phone_for_folder}_{uuid.uuid4().hex[:8]}{ext}"
                    selfie_path = os.path.join(USER_SELFIE_STORAGE_DIR, filename)
                    shutil.copy(temp_path, selfie_path)
                    print(f"Selfie copied and stored as {selfie_path}")
                except Exception as e:
                    print(f"Error copying selfie: {e}. Please try again or check permissions.")
                    selfie_path = None # Reset on error
            else:
                print("Invalid file path. Please try again.")
        elif selfie_input_mode == '2':
            print("Preparing webcam...")
            selfie_path = capture_selfie_from_webcam(sanitized_phone_for_folder)
            if not selfie_path:
                print("Webcam capture failed or was cancelled. Please try again or choose another option.")
        else:
            print("Invalid choice. Please enter 1 or 2.")

    if not selfie_path:
        print("No selfie provided. Exiting program.")
        return

    # 3. Load and encode the selfie
    print("\n--- Processing Selfie ---")
    selfie_encoding = load_and_encode_face(selfie_path)

    if selfie_encoding is None:
        print("Could not process the selfie (no face found or error). Exiting.")
        return

    print("Selfie processed successfully.")

    # 4. Find matching photos
    print("\n--- Searching for Your Photos ---")
    if not os.listdir(SOURCE_PHOTOS_DIR):
        print(f"The source photo directory '{SOURCE_PHOTOS_DIR}' is empty. Please add photos to search.")
        return

    matched_photos = find_matching_photos(selfie_encoding, SOURCE_PHOTOS_DIR)

    if not matched_photos:
        print("\nNo matching photos found in our collection.")
    else:
        print(f"\nFound {len(matched_photos)} potential match(es).")
        # 5. Create folder and copy photos
        user_folder_path = create_user_folder_and_copy_photos(
            sanitized_phone_for_folder,
            matched_photos,
            OUTPUT_BASE_DIR
        )
        print(f"\nAll your matched photos have been organized in: {user_folder_path}")
        print("You can now access this folder on your computer.")

    print("\n------------------------------------")
    print("Thank you for using the Photo Finder Service!")

if __name__ == "__main__":
    main()