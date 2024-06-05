import cv2
import numpy as np

def find_midpoints(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to separate white strip from the rest
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours of the white strip
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Calculate midpoints along the white strip
    midpoints = []
    for contour in contours[0]:
      cx, cy = contour[0]
      midpoints.append([cx, cy])
      '''
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Ensure non-zero area
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            midpoints.append((cx, cy))
      '''
    return np.array(midpoints)

# Provide the path to your PNG image file
image_path = r"T:\Code.Thesis\AutonomousRacing\f1tenth_racetracks\Monaco\Monaco.png"

# Call the function to find midpoints
midpoints = find_midpoints(image_path)
print(midpoints.shape)
# Print the calculated midpoints
for point in midpoints:
    print(point)
