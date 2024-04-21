import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

class TFLiteImageClassifier:

    def __init__(self, model_path, labels_path, cam_index=0):
        self.MODEL_PATH = model_path
        self.LABELS_PATH = labels_path
        self.LABEL_ARR = []

        self._class_string = "null"


        self.load_model()
        self.initialize_video_capture(cam_index)

    def load_model(self):
        self.interpreter = tflite.Interpreter(model_path=self.MODEL_PATH)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]

    def initialize_video_capture(self, cam_index):
        self.cap = cv2.VideoCapture(cam_index)

    def convert_label_to_array(self):
        label_list = []
        with open(self.LABELS_PATH, 'r') as file:
            for line in file:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    _, value = parts
                    label_list.append(value)
        self.LABEL_ARR = label_list

    def get_label_index(self, class_index):
        return self.LABEL_ARR[class_index]

    def run_inference(self, frame):
        img = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))
        img = img.reshape((1,) + img.shape)  # Add batch dimension
        img = img.astype(np.uint8)  # Ensure the input data type is UINT8

        # Set the input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], img)

        # Run inference
        self.interpreter.invoke()

        # Get the output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = output_data[0]

        # Get the class with the highest probability
        class_index = np.argmax(predictions)
        confidence = predictions[class_index]

        return class_index, confidence

    def display_results(self, frame, class_index, confidence):
        class_label = f"Class: {class_index}, Confidence: {confidence:.2f}"
        cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cls_string = self.get_label_index(class_index)

        cv2.putText(frame, cls_string, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Webcam Feed', frame)
        return cls_string
    
    def getClassResults(self):
        return self._class_string
    
    def setWindowProp(self):
        cv2.namedWindow('Webcam Feed', cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty('Webcam Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)        

    def capture_and_classify(self):

        while True:
            ret, frame = self.cap.read()

            class_index, confidence = self.run_inference(frame)

            self._class_string = self.display_results(frame, class_index, confidence)

            # print(self._class_string)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    model_path = '/home/rifay/tfLite_imgClass/tfLiteOwn/MMS_sorting_test_tflite_quantized/model.tflite'
    labels_path = '/home/rifay/tfLite_imgClass/tfLiteOwn/MMS_sorting_test_tflite_quantized/labels.txt'

    # Enter Camera Index
    camera_index = 2

    image_classifier = TFLiteImageClassifier(model_path, labels_path, camera_index)
    image_classifier.convert_label_to_array()

    image_classifier.setWindowProp()
    # Infinite Loop:
    image_classifier.capture_and_classify() 

if __name__ == "__main__":
    main()
