import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
import torch
import numpy as np

# --- SAHI Imports ---
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    print("❌ SAHI not installed. Please run: pip install sahi")
    SAHI_AVAILABLE = False
    from ultralytics import YOLO # Fallback

# =============================================================================
#  MERGED CONFIGURATION (Using settings from Script 1 for best results)
# =============================================================================
SAHI_CONFIG = {
    # Path to your model
    "model_path": "best.pt", 

    # --- Slicing Parameters (From Script 1) ---
    # Smaller slices (256) help avoid detecting large shadows/empty spaces
    "slice_height": 256,
    "slice_width": 256,
    "overlap_height_ratio": 0.2,
    "overlap_width_ratio": 0.2,

    # --- Confidence & Logic ---
    "model_confidence_threshold": 0.40,  # Minimum to see anything
    "certainty_threshold": 0.45,         # Below this = "Unknown"
    
    # NMS (Removes overlapping boxes)
    "iou_threshold": 0.5,
    "postprocess_class_agnostic": True   
}
# =============================================================================


# --- Setup Camera ---
cap = cv2.VideoCapture(1) # Try 0 if 1 doesn't work

# --- Global State Variables ---
is_paused = False       
last_frame = None       
camera_label = None     
model = None            

# --- Load Model (Run once at startup) ---
print("--- Loading AI Model... Please wait ---")
try:
    # Check for GPU
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.empty_cache()
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ CUDA not available, using CPU.")
    
    if SAHI_AVAILABLE:
        # Initialize SAHI Model
        model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=SAHI_CONFIG["model_path"],
            confidence_threshold=SAHI_CONFIG["model_confidence_threshold"], 
            device=device,
        )
        print("✅ SAHI + YOLO Model loaded successfully!")
    else:
        # Fallback to standard YOLO if SAHI is missing
        model = YOLO(SAHI_CONFIG["model_path"])
        model.to(device)
        print("✅ Standard YOLO Model loaded (SAHI missing).")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"Make sure '{SAHI_CONFIG['model_path']}' is in the folder.")


# --- Detection Function (Logic from Script 1) ---
def detect_objects(input_image):
    """
    Takes a cv2 image, runs SAHI inference using Script 1's logic,
    and returns the annotated image.
    """
    global model
    
    if model is None:
        print("Error: Model not loaded.")
        return input_image

    annotated_img = input_image.copy()

    # Convert BGR (OpenCV) to RGB for SAHI processing
    rgb_image = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    try:
        if SAHI_AVAILABLE:
            # --- Perform Sliced Prediction ---
            result = get_sliced_prediction(
                rgb_image,
                model,
                slice_height=SAHI_CONFIG["slice_height"],
                slice_width=SAHI_CONFIG["slice_width"],
                overlap_height_ratio=SAHI_CONFIG["overlap_height_ratio"],
                overlap_width_ratio=SAHI_CONFIG["overlap_width_ratio"],
                postprocess_type="NMS",
                postprocess_match_metric="IOS",
                postprocess_match_threshold=SAHI_CONFIG["iou_threshold"],
                postprocess_class_agnostic=SAHI_CONFIG["postprocess_class_agnostic"]
            )

            object_prediction_list = result.object_prediction_list
            print(f"✅ SAHI Complete: Found {len(object_prediction_list)} objects.")

            if len(object_prediction_list) > 0:
                for prediction in object_prediction_list:
                    # 1. Get Score & Box
                    score = prediction.score.value
                    bbox = prediction.bbox
                    x_min, y_min, x_max, y_max = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
                    
                    # 2. Apply "Unknown" Logic (From Script 1)
                    label_name = prediction.category.name
                    
                    # Green for Known
                    color = (0, 255, 0) 

                    if score < SAHI_CONFIG["certainty_threshold"]:
                        label_name = "Unknown"
                        # Orange for Unknown (Script 1 color)
                        color = (0, 165, 255) 
                        # Note: OpenCV uses BGR, so (0, 165, 255) is Orange/Gold

                    # 3. Draw Rectangle
                    start_point = (int(x_min), int(y_min))
                    end_point = (int(x_max), int(y_max))
                    
                    cv2.rectangle(annotated_img, start_point, end_point, color, 2)

                    # 4. Draw Label
                    label_text = f"{label_name}: {score:.2f}"
                    text_loc = (int(x_min), int(y_min) - 10 if int(y_min) - 10 > 10 else int(y_min) + 10)
                    
                    cv2.putText(annotated_img, label_text, text_loc, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                print("No objects detected.")

            return annotated_img

        else:
            # --- STANDARD YOLO FALLBACK (If SAHI fails) ---
            results = model(annotated_img, iou=0.4, conf=0.4, agnostic_nms=True)
            if results and results[0].boxes:
                detections = results[0].boxes
                for box in detections:
                    xyxy = box.xyxy[0].int().tolist()
                    cv2.rectangle(annotated_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            return annotated_img

    except Exception as e:
        print(f"Error during inference: {e}")
        return input_image


# --- GUI Functionality (From Script 2) ---

def logo_button_clicked():
    """Gallery Logic"""
    save_folder = "processed_images"
    
    if not os.path.exists(save_folder):
        print("Gallery Error: No processed images folder found.")
        return

    try:
        images = [f for f in os.listdir(save_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images.sort(reverse=True) # Newest first
    except Exception as e:
        print(f"Gallery Error: {e}")
        return

    if not images:
        print("Gallery Error: No images found in the folder.")
        return

    gallery_window = tk.Toplevel()
    gallery_window.title("Gallery")
    gallery_window.geometry("600x700") 
    gallery_window.configure(bg="#FFD782")
    gallery_window.overrideredirect(True)

    # Top Bar
    top_bar = tk.Frame(gallery_window, bg="#FFD782")
    top_bar.pack(side="top", fill="x", padx=15, pady=15)
    
    btn_back = tk.Button(top_bar, text="← Back", command=gallery_window.destroy, font=("Arial", 11, "bold"), bg="white", borderwidth=0)
    btn_back.pack(side="left")

    # Image Area
    gallery_label = tk.Label(gallery_window, bg="#FFD782")
    gallery_label.pack(side="top", pady=10, expand=True)

    # Controls
    controls_frame = tk.Frame(gallery_window, bg="#FFD782")
    controls_frame.pack(side="bottom", pady=20)
    name_label = tk.Label(gallery_window, text="", bg="#FFD782", font=("Arial", 10, "bold"))
    name_label.pack(side="bottom", pady=5)

    current_idx = [0] 

    def show_image():
        filename = images[current_idx[0]]
        filepath = os.path.join(save_folder, filename)
        try:
            pil_img = Image.open(filepath)
            pil_img.thumbnail((550, 500)) 
            gallery_img = ImageTk.PhotoImage(pil_img)
            gallery_label.configure(image=gallery_img)
            gallery_label.image = gallery_img
            name_label.configure(text=f"{filename}  ({current_idx[0] + 1}/{len(images)})")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    def next_image():
        current_idx[0] = (current_idx[0] + 1) % len(images)
        show_image()

    def prev_image():
        current_idx[0] = (current_idx[0] - 1 + len(images)) % len(images)
        show_image()

    tk.Button(controls_frame, text="<< Prev", command=prev_image, font=("Arial", 12), bg="white").pack(side="left", padx=20)
    tk.Button(controls_frame, text="Next >>", command=next_image, font=("Arial", 12), bg="white").pack(side="left", padx=20)

    show_image()

def upload_image_clicked():
    global is_paused, last_frame
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    if file_path:
        is_paused = True
        print(f"Input: Loaded image from {file_path}")
        loaded_frame = cv2.imread(file_path)
        if loaded_frame is not None:
            last_frame = loaded_frame
            cv2image = cv2.cvtColor(loaded_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img = img.resize((564, 520))
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.configure(image=imgtk)
            camera_label.image = imgtk 
        else:
            print("Error: Could not read the image file.")

def capture_photo_clicked():
    global is_paused
    if not is_paused:
        is_paused = True
        print("Capture: Feed Paused (Frame Captured)")
    else:
        is_paused = False
        print("Reset: Camera Feed Resumed")
        update_camera_feed(camera_label)

def run_model_clicked():
    global last_frame
    
    if is_paused and last_frame is not None:
        print("Run Model: Processing image...")
        
        # 1. RUN DETECTION
        annotated_frame = detect_objects(last_frame)
        
        # 2. SAVE LOGIC
        save_folder = "processed_images"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_folder, f"berry_scan_{timestamp}.jpg")
        
        cv2.imwrite(filename, annotated_frame)
        print(f"✅ Saved result to: {filename}")
        
        # 3. DISPLAY RESULT
        cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        img = img.resize((564, 520))
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.configure(image=imgtk)
        camera_label.image = imgtk
        
    else:
        print("Please 'Capture' a photo or 'Upload' an image before running the model.")

def update_camera_feed(label):
    global is_paused, last_frame
    if is_paused:
        return
    ret, frame = cap.read()
    if ret:
        last_frame = frame
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        img = img.resize((564, 520)) 
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    label.after(10, lambda: update_camera_feed(label))   

# --- Main Application Setup (GUI from Script 2) ---
def create_berryscan_interface():
    global camera_label
    
    root = tk.Tk()
    root.title("BerryScan")
    root.geometry("1024x600")
    root.overrideredirect(True) # No Title Bar
    root.configure(bg="#FFD782")

    title_font = tkfont.Font(family="Arial", size=46, weight="bold")
    button_font = tkfont.Font(family="Arial", size=12, weight="bold")

    # 1. LOGO BUTTON
    image_path = r"E:\Personal Files\ThesisGUI_Final\Button.png"
    try:
        icon_image = tk.PhotoImage(file=image_path)
        logo_button = tk.Button(root, image=icon_image, bg="#FFD782", command=logo_button_clicked, borderwidth=0, cursor="hand2")
        logo_button.image = icon_image 
    except Exception:
        # Fallback if image not found
        logo_button = tk.Button(root, text="[Gallery]", bg="black", fg="white", command=logo_button_clicked)

    logo_button.place(x=40, y=40, width=96, height=96)

    # 2. LABELS & CONTROLS
    tk.Label(root, text="Berry", bg="#FFD782", fg="#E82C2A", font=title_font).place(x=40, y=140)
    tk.Label(root, text="Scan", bg="#FFD782", fg="black", font=title_font).place(x=205, y=140)

    # Buttons
    tk.Button(root, text="CAPTURE PHOTO", bg="#262626", fg="white", font=button_font, command=capture_photo_clicked, borderwidth=0).place(x=40, y=250, width=340, height=70)
    tk.Button(root, text="RUN MODEL", bg="#C91B1A", fg="white", font=button_font, command=run_model_clicked, borderwidth=0).place(x=40, y=340, width=340, height=70)
    tk.Button(root, text="UPLOAD IMAGE", bg="#005A9C", fg="white", font=button_font, command=upload_image_clicked, borderwidth=0).place(x=40, y=430, width=340, height=70)

    # 3. CAMERA FEED
    camera_label = tk.Label(root, bg="#F48057")
    camera_label.place(x=420, y=40, width=564, height=520)
    
    update_camera_feed(camera_label)

    # Escape to Exit Logic
    def on_closing(event=None):
        cap.release()
        root.destroy()

    root.bind("<Escape>", on_closing)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    create_berryscan_interface()