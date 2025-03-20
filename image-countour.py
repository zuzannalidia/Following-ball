import cv2
import numpy as np
import os


def track_ball(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    h, w = image.shape[:2]
    if h > w:
        if h > 800:
            s = (1 - (800 / h)) * (-1)
            w = w + int(w * (s))
            h = h + int(h * (s))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        if w > 800:
            s = (1 - (800 / w)) * (-1)
            w = w + int(w * (s))
            h = h + int(h * (s))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    cv2.namedWindow('Original')
    cv2.namedWindow('HSV Controls')
    cv2.namedWindow('Mask')
    cv2.namedWindow('Filtered Result')
    cv2.namedWindow('Result with Marker')

    cv2.createTrackbar('Low H', 'HSV Controls', 0, 179, lambda x: None)
    cv2.createTrackbar('High H', 'HSV Controls', 179, 179, lambda x: None)
    cv2.createTrackbar('Low S', 'HSV Controls', 0, 255, lambda x: None)
    cv2.createTrackbar('High S', 'HSV Controls', 255, 255, lambda x: None)
    cv2.createTrackbar('Low V', 'HSV Controls', 0, 255, lambda x: None)
    cv2.createTrackbar('High V', 'HSV Controls', 255, 255, lambda x: None)
    cv2.createTrackbar('Kernel Size', 'HSV Controls', 5, 20, lambda x: None)

    cv2.setTrackbarPos('Low H', 'HSV Controls', 0)
    cv2.setTrackbarPos('High H', 'HSV Controls', 20)
    cv2.setTrackbarPos('Low S', 'HSV Controls', 100)
    cv2.setTrackbarPos('High S', 'HSV Controls', 255)
    cv2.setTrackbarPos('Low V', 'HSV Controls', 100)
    cv2.setTrackbarPos('High V', 'HSV Controls', 255)

    while True:
        # Get current trackbar values
        low_h = cv2.getTrackbarPos('Low H', 'HSV Controls')
        high_h = cv2.getTrackbarPos('High H', 'HSV Controls')
        low_s = cv2.getTrackbarPos('Low S', 'HSV Controls')
        high_s = cv2.getTrackbarPos('High S', 'HSV Controls')
        low_v = cv2.getTrackbarPos('Low V', 'HSV Controls')
        high_v = cv2.getTrackbarPos('High V', 'HSV Controls')
        kernel_size = cv2.getTrackbarPos('Kernel Size', 'HSV Controls')
        kernel_size = max(1, kernel_size)

        cv2.imshow('Original', image)

        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array([low_h, low_s, low_v])
        upper = np.array([high_h, high_s, high_v])
        mask = cv2.inRange(hsv_frame, lower, upper)
        cv2.imshow('Mask', mask)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

        result = cv2.bitwise_and(image, image, mask=mask_cleaned)
        cv2.imshow('Filtered Result', result)

        try:
            contours_output = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_output) == 2:
                contours, _ = contours_output
            else:
                _, contours, _ = contours_output

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                M = cv2.moments(largest_contour)

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    result_with_marker = image.copy()

                    cv2.drawMarker(result_with_marker, (cx, cy),
                                   color=(0, 255, 0),
                                   markerType=cv2.MARKER_CROSS,
                                   markerSize=20,
                                   thickness=2)

                    cv2.drawContours(result_with_marker, [largest_contour], -1, (0, 0, 255), 2)

                    cv2.imshow('Result with Marker', result_with_marker)
                else:
                    cv2.imshow('Result with Marker', image)
            else:
                cv2.imshow('Result with Marker', image)
        except Exception as e:
            print(f"Error finding contours: {e}")
            cv2.imshow('Result with Marker', image)

        print(f"Current HSV Range: [{low_h},{low_s},{low_v}] to [{high_h},{high_s},{high_v}]", end='\r')


        key = cv2.waitKey(100)
        if key == 27:
            print("\nFinal HSV Range:")
            print(f"Lower HSV: [{low_h}, {low_s}, {low_v}]")
            print(f"Upper HSV: [{high_h}, {high_s}, {high_v}]")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    ball_image_path = r"C:\Users\zuzan\Desktop\Semestr6\WMA\PRO1\ball.png"
    track_ball(ball_image_path)