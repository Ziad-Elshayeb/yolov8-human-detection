# Load needed libraries
import cv2
import time
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model (adjust this to your model path)
model = YOLO('human_detection_model.pt')  # Replace with the path to your YOLOv8 model

# Define the video file path (adjust this to your video path)
video_path = 'test video.mp4'  # Replace with the path to your video file

# Open the video file
video = cv2.VideoCapture(video_path)

# Get video properties
video_fps = video.get(cv2.CAP_PROP_FPS)  # Get the original FPS of the video
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_delay = int(1000 / (video_fps))  # Delay between frames in milliseconds
print(f"Video FPS: {video_fps}, Total Frames: {total_frames}, Frame Delay: {frame_delay} ms")

# Check if the video opened successfully
if not video.isOpened():
    print("Error opening video file")
    exit()

# Variables to keep track of time
total_inference_time = 0.0
frame_count = 0

# Start timer to calculate total video processing time
start_total_time = time.time()

# Loop over video frames
while True:
    ret, frame = video.read()
    if not ret:
        break  # Exit if no more frames

    # Start the inference timer for each frame
    start_time = time.time()

    # Perform inference on the frame
    results = model(frame)  # Run YOLOv8 inference on the frame

    # End the inference timer for each frame
    end_time = time.time()

    # Calculate inference time for this frame
    inference_time = end_time - start_time
    total_inference_time += inference_time
    frame_count += 1

    # Display progress
    print(f"Frame {frame_count}/{total_frames} processed. Inference time: {inference_time:.4f} seconds")

    # Iterate over results and draw bounding boxes and labels on the frame
    for result in results[0].boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        confidence = result.conf[0]
        class_id = int(result.cls[0])
        label = model.names[class_id]  # Get the label for the class

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the label and confidence score
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Frame", frame)

    # Add a delay to maintain the original frame rate of the video
    # you can use the frame_delay variable above but will make the video slower and take more time to finish
    if cv2.waitKey(12) & 0xFF == ord('q'): 
        break

# End timer to calculate total video processing time
end_total_time = time.time()

# Calculate total processing time for the video
total_video_time = end_total_time - start_total_time

# Calculate average FPS during inference
average_inference_fps = frame_count / total_inference_time

print(f"\nTotal frames processed: {frame_count}")
print(f"Total inference time: {total_inference_time:.2f} seconds")
print(f"Average inference FPS: {average_inference_fps:.2f}")
print(f"Total video processing time: {total_video_time:.2f} seconds")  # Total time to process the video

# Release the video and close windows
video.release()
cv2.destroyAllWindows()
