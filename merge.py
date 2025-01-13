import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Step 1: Preprocess Image - Edge Detection (Canny)
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection with custom thresholds
    edges = cv2.Canny(blurred, 50, 150)
    
    return image, gray, blurred, edges

# Step 2: Let the user select ROI manually
def select_roi(image):
    # Let the user select the region of interest (ROI) interactively
    roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return roi

# Step 3: Apply Hough Transform (manual implementation)
def hough_transform(edges, theta_res=1, rho_res=1):
    height, width = edges.shape
    diag_len = int(np.sqrt(width**2 + height**2))  # Diagonal length for rho range
    rho_max = diag_len
    rho_min = -diag_len
    
    num_thetas = int(180 / theta_res)
    num_rhos = int(2 * rho_max / rho_res)
    
    accumulator = np.zeros((num_rhos, num_thetas), dtype=int)
    
    # Iterate through all the edge pixels
    for y in range(height):
        for x in range(width):
            if edges[y, x] == 255:  # Only consider edge pixels
                for theta_idx in range(num_thetas):
                    theta = np.deg2rad(theta_idx * theta_res)
                    rho = x * np.cos(theta) + y * np.sin(theta)
                    rho_idx = int((rho + rho_max) / rho_res)
                    accumulator[rho_idx, theta_idx] += 1
    
    return accumulator, rho_min, rho_max, num_thetas, num_rhos

# Step 4: Detect lines from the accumulator
def detect_lines(accumulator, threshold=50):  # Lowered threshold to detect more lines
    lines = []
    num_rhos, num_thetas = accumulator.shape
    
    # Iterate through the accumulator and detect peaks
    for rho_idx in range(num_rhos):
        for theta_idx in range(num_thetas):
            if accumulator[rho_idx, theta_idx] > threshold:  # Reduced threshold
                rho = rho_idx - num_rhos / 2
                theta = np.deg2rad(theta_idx)
                lines.append((rho, theta))
    
    return lines

# Step 5: Draw the detected lines on the original image (in red)
def draw_lines_on_image(image, lines, x_offset, y_offset):
    # Draw red lines on the original image
    for rho, theta in lines:
        # Convert rho and theta to line coordinates (x1, y1) and (x2, y2)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        x1 = int(rho * cos_theta - image.shape[0] * sin_theta)
        y1 = int(rho * sin_theta + image.shape[0] * cos_theta)
        x2 = int(rho * cos_theta + image.shape[1] * sin_theta)
        y2 = int(rho * sin_theta - image.shape[1] * cos_theta)
        
        # Translate coordinates based on ROI offset
        x1 += x_offset
        y1 += y_offset
        x2 += x_offset
        y2 += y_offset
        
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines (BGR format)
    return image

# Main function to process the image
def lane_detection():
    # Step 1: Open file dialog to select an image
    Tk().withdraw()  # Hides the root Tkinter window
    image_path = askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if not image_path:
        print("No image selected.")
        return
    
    # Step 2: Preprocess the image (Canny edge detection)
    image, gray, blurred, edges = preprocess_image(image_path)
    
    # Step 3: Let the user select ROI manually
    roi = select_roi(image)
    x, y, w, h = roi
    
    # Step 4: Crop the Canny edge image to the selected ROI
    roi_edges = edges[y:y+h, x:x+w]
    
    # Step 5: Apply Hough Transform to detect lines in the ROI
    accumulator, rho_min, rho_max, num_thetas, num_rhos = hough_transform(roi_edges)
    
    # Step 6: Detect lines from the accumulator
    lines = detect_lines(accumulator, threshold=50)  # Adjusted threshold for more lines
    
    # Step 7: Draw the detected lines (red) on a copy of the original image (with ROI offset)
    image_with_lines = image.copy()
    image_with_lines = draw_lines_on_image(image_with_lines, lines, x, y)
    
    # Step 8: Crop the final image with red lane lines to the ROI area
    roi_with_lines = image_with_lines[y:y+h, x:x+w]
    
    # Step 9: Create the final output by pasting the ROI with lines back onto the original image
    final_image = image.copy()
    final_image[y:y+h, x:x+w] = roi_with_lines  # Paste the cropped region with lane lines into the original image
    
    # Display the results in a multi-panel format
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    
    plt.subplot(3, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    
    plt.subplot(3, 2, 3)
    plt.imshow(blurred, cmap='gray')
    plt.title('Blurred Image')
    
    plt.subplot(3, 2, 4)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection (Canny)')
    
    plt.subplot(3, 2, 5)
    plt.imshow(roi_edges, cmap='gray')
    plt.title('Edge Detection (Canny) - ROI')
    
    plt.subplot(3, 2, 6)
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.title('Lane Markers Highlighted in Red on Original Image')
    
    plt.tight_layout()
    plt.show()

# Run the lane detection
lane_detection()
