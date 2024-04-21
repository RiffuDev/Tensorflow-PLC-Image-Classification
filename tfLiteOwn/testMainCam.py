import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the TFLite model
MODEL_PATH = '/home/rifay/tfLite_imgClass/tfLiteOwn/MMS_sorting_test_tflite_quantized/model.tflite'  # Replace with the actual path to your .tflite model
LABELS_PATH = '/home/rifay/tfLite_imgClass/tfLiteOwn/MMS_sorting_test_tflite_quantized/labels.txt'  # Replace with the actual path to your labels.txt file

LABEL_ARR = []

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

#======================================

def convertLabelToArray(labels_file_path):

    # Read labels from the file and create a list
    label_list = []
    with open(labels_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                _, value = parts
                label_list.append(value)

    # Display the resulting list
    print(label_list)

    return label_list

def getLabelIndex(classIndex):
    global LABEL_ARR
    temp = LABEL_ARR[classIndex]
    return temp


#======================================

def cvLoop():

    # enabling full screen
    cv2.namedWindow('Webcam Feed', cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty('Webcam Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Preprocess the frame for inference
        img = cv2.resize(frame, (input_shape[1], input_shape[0]))
        img = img.reshape((1,) + img.shape)  # Add batch dimension
        img = img.astype(np.uint8)  # Ensure the input data type is UINT8

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions = output_data[0]

        # Get the class with the highest probability
        class_index = np.argmax(predictions)
        confidence = predictions[class_index]

        # Display the class label and confidence on the frame
        class_label = f"Class: {class_index}, Confidence: {confidence:.2f}"
        cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        clsString = getLabelIndex(class_index)

        cv2.putText(frame, clsString, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Webcam Feed', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()


def cvMain():

    global LABEL_ARR
    LABEL_ARR = convertLabelToArray(LABELS_PATH)

    cvLoop()


# main()