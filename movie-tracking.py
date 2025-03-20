import cv2
import numpy as np


def track_ball_in_video(input_video_path, output_video_path):
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file at {input_video_path}")
        return

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    output = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'MJPG'),
        fps,
        (frame_width, frame_height)
    )

    lower_hsv = np.array([0, 92, 147])
    upper_hsv = np.array([179, 255, 255])

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    cv2.namedWindow("Ball Tracking", cv2.WINDOW_NORMAL)

    counter = 1
    while True:
        success, frame = video.read()
        if not success:
            break

        print(f'Processing frame {counter} of {total_frames}')

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_display = frame.copy()

        if contours:
            min_contour_area = 50
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

            cv2.drawContours(contour_display, valid_contours, -1, (255, 0, 0), 1)

            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)

                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    cv2.drawContours(contour_display, [largest_contour], -1, (0, 0, 255), 2)

                    cv2.drawMarker(contour_display, (cx, cy),
                                   color=(0, 255, 0),
                                   markerType=cv2.MARKER_CROSS,
                                   markerSize=20,
                                   thickness=2)

        cv2.imshow("Ball Tracking", contour_display)


        output.write(contour_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        counter += 1

    video.release()
    output.release()
    cv2.destroyAllWindows()

    print(f"Processing complete. Output saved to {output_video_path}")


if __name__ == "__main__":
    input_path = r'C:\Users\zuzan\Desktop\Semestr6\WMA\PRO1\movingball.mp4'
    output_path = 'ball_tracking_result.avi'
    track_ball_in_video(input_path, output_path)