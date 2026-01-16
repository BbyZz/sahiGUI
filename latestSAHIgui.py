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
#  CONFIGURATION
# =============================================================================
SAHI_CONFIG = {
    "model_path": "ckbest.pt", 
    "slice_height": 256,
    "slice_width": 256,
    "overlap_height_ratio": 0.2,
    "overlap_width_ratio": 0.2,
    "model_confidence_threshold": 0.60,  
    "certainty_threshold": 0.60,         
    "iou_threshold": 0.5,
    "postprocess_class_agnostic": True   
}

# --- Setup Camera ---
cap = cv2.VideoCapture(2) # Try 0 if 1 doesn't work

# --- Global State Variables ---
is_paused = False       
last_frame = None       
camera_label = None     
model = None            

# --- Load Model ---
print("--- Loading AI Model... Please wait ---")
try:
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.empty_cache()
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ CUDA not available, using CPU.")
    
    if SAHI_AVAILABLE:
        model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=SAHI_CONFIG["model_path"],
            confidence_threshold=SAHI_CONFIG["model_confidence_threshold"], 
            device=device,
        )
        print("✅ SAHI + YOLO Model loaded successfully!")
    else:
        model = YOLO(SAHI_CONFIG["model_path"])
        model.to(device)
        print("✅ Standard YOLO Model loaded (SAHI missing).")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"Make sure '{SAHI_CONFIG['model_path']}' is in the folder.")


# --- Detection Function (Updated with Tally) ---
def detect_objects(input_image):
    global model
    
    if model is None:
        print("Error: Model not loaded.")
        return input_image

    annotated_img = input_image.copy()
    rgb_image = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Initialize Tally Dictionary
    label_counts = {}

    try:
        if SAHI_AVAILABLE:
            # --- Perform Prediction ---
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
                    # 1. Get Data
                    score = prediction.score.value
                    bbox = prediction.bbox
                    x_min, y_min, x_max, y_max = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
                    
                    # 2. Logic & Tally
                    label_name = prediction.category.name
                    color = (0, 255, 0) # Green

                    if score < SAHI_CONFIG["certainty_threshold"]:
                        label_name = "Unknown"
                        color = (0, 165, 255) # Orange

                    # Update Tally
                    label_counts[label_name] = label_counts.get(label_name, 0) + 1

                    # 3. Draw Box & Label
                    cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    label_text = f"{label_name} {score:.2f}"
                    text_loc = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
                    cv2.putText(annotated_img, label_text, text_loc, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # --- DRAW TALLY SUMMARY ON IMAGE ---
                # Define starting position for the tally list
                start_x, start_y = 10, 30
                
                # Draw a semi-transparent background for the text
                overlay = annotated_img.copy()
                # Calculate height needed: 30px per line
                box_h = len(label_counts) * 30 + 10 
                cv2.rectangle(overlay, (5, 5), (200, box_h), (0, 0, 0), -1)
                alpha = 0.6 # Transparency factor
                cv2.addWeighted(overlay, alpha, annotated_img, 1 - alpha, 0, annotated_img)

                # Write the text
                print(f"📊 Tally: {label_counts}") # Print to console
                for i, (label, count) in enumerate(label_counts.items()):
                    text = f"{label}: {count}"
                    y_pos = start_y + (i * 30)
                    cv2.putText(annotated_img, text, (start_x, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            else:
                print("No objects detected.")

            return annotated_img

        else:
            # --- STANDARD YOLO FALLBACK ---
            results = model(annotated_img, iou=0.4, conf=0.4, agnostic_nms=True)
            if results and results[0].boxes:
                detections = results[0].boxes
                for box in detections:
                    xyxy = box.xyxy[0].int().tolist()
                    class_id = int(box.cls[0].item())
                    label = model.names[class_id]
                    
                    # Tally
                    label_counts[label] = label_counts.get(label, 0) + 1
                    
                    cv2.rectangle(annotated_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                
                # Draw Tally for Fallback
                y_offset = 30
                for label, count in label_counts.items():
                    cv2.putText(annotated_img, f"{label}: {count}", (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    y_offset += 35
                    
            return annotated_img

    except Exception as e:
        print(f"Error during inference: {e}")
        return input_image


# --- GUI Functionality ---

def logo_button_clicked():
    save_folder = "processed_images"
    if not os.path.exists(save_folder): return
    
    try:
        images = [f for f in os.listdir(save_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images.sort(reverse=True)
    except: return

    if not images: return

    gallery_window = tk.Toplevel()
    gallery_window.title("Gallery")
    gallery_window.geometry("600x700") 
    gallery_window.configure(bg="#FFD782")
    gallery_window.overrideredirect(True)

    top_bar = tk.Frame(gallery_window, bg="#FFD782")
    top_bar.pack(side="top", fill="x", padx=15, pady=15)
    btn_back = tk.Button(top_bar, text="← Back", command=gallery_window.destroy, font=("Arial", 11, "bold"), bg="white", borderwidth=0)
    btn_back.pack(side="left")

    gallery_label = tk.Label(gallery_window, bg="#FFD782")
    gallery_label.pack(side="top", pady=10, expand=True)

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
        except: pass

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
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")])
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

def capture_photo_clicked():
    global is_paused
    if not is_paused:
        is_paused = True
        print("Capture: Feed Paused")
    else:
        is_paused = False
        print("Reset: Feed Resumed")
        update_camera_feed(camera_label)

def run_model_clicked():
    global last_frame
    if is_paused and last_frame is not None:
        print("Run Model: Processing...")
        annotated_frame = detect_objects(last_frame)
        
        save_folder = "processed_images"
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_folder, f"berry_scan_{timestamp}.jpg")
        
        cv2.imwrite(filename, annotated_frame)
        print(f"✅ Saved: {filename}")
        
        cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        img = img.resize((564, 520))
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.configure(image=imgtk)
        camera_label.image = imgtk
    else:
        print("Please Capture/Upload first.")

def update_camera_feed(label):
    global is_paused, last_frame
    if is_paused: return
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

# --- Main Setup ---
def create_berryscan_interface():
    global camera_label
    root = tk.Tk()
    root.title("BerryScan")
    root.geometry("1024x600")
    root.overrideredirect(True) 
    root.configure(bg="#FFD782")

    title_font = tkfont.Font(family="Arial", size=46, weight="bold")
    button_font = tkfont.Font(family="Arial", size=12, weight="bold")

    image_path = r"E:\Personal Files\ThesisGUI_Final\Button.png"
    try:
        icon_image = tk.PhotoImage(file=image_path)
        logo_button = tk.Button(root, image=icon_image, bg="#FFD782", command=logo_button_clicked, borderwidth=0, cursor="hand2")
        logo_button.image = icon_image 
    except:
        logo_button = tk.Button(root, text="[Gallery]", bg="black", fg="white", command=logo_button_clicked)
    logo_button.place(x=40, y=40, width=96, height=96)

    tk.Label(root, text="Berry", bg="#FFD782", fg="#E82C2A", font=title_font).place(x=40, y=140)
    tk.Label(root, text="Scan", bg="#FFD782", fg="black", font=title_font).place(x=205, y=140)

    tk.Button(root, text="CAPTURE PHOTO", bg="#262626", fg="white", font=button_font, command=capture_photo_clicked, borderwidth=0).place(x=40, y=250, width=340, height=70)
    tk.Button(root, text="RUN MODEL", bg="#C91B1A", fg="white", font=button_font, command=run_model_clicked, borderwidth=0).place(x=40, y=340, width=340, height=70)
    tk.Button(root, text="UPLOAD IMAGE", bg="#005A9C", fg="white", font=button_font, command=upload_image_clicked, borderwidth=0).place(x=40, y=430, width=340, height=70)

    camera_label = tk.Label(root, bg="#F48057")
    camera_label.place(x=420, y=40, width=564, height=520)
    
    update_camera_feed(camera_label)

    def on_closing(event=None):
        cap.release()
        root.destroy()
    root.bind("<Escape>", on_closing)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    create_berryscan_interface()