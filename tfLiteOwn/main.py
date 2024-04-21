
import sys
sys.path.append(r'/home/rifay/tfLite_imgClass/tfLiteOwn')

import threading

import time

import blah
import syncOpcua

model_path = '/home/rifay/tfLite_imgClass/tfLiteOwn/MMS_sorting_test_tflite_quantized/model.tflite'
labels_path = '/home/rifay/tfLite_imgClass/tfLiteOwn/MMS_sorting_test_tflite_quantized/labels.txt'

# Enter Camera Index
camera_index = 2

t1Flag = True

def runCV(image_classifier):

    image_classifier.convert_label_to_array()
    image_classifier.setWindowProp()
    # Infinite Loop:
    image_classifier.capture_and_classify() 



def handleResults(image_classifier):

    # flg, client = syncOpcua.connect("opc.tcp://192.168.11.47:4840")

    while(t1Flag):

        

        r = image_classifier.getClassResults()
        print(f"RES:{r}")

        if(r == "Box_Steel"):

            print("Thuku Thuku")

            # syncOpcua.pickCyl(client)
            syncOpcua.connect("opc.tcp://192.168.11.47:4840")

        time.sleep(1)

def main():
    
    global t1Flag

    image_classifier = blah.TFLiteImageClassifier(model_path, labels_path, camera_index)

    t1 = threading.Thread(target=handleResults, args=(image_classifier,))

    t1.start()
    runCV(image_classifier)
    t1Flag = False
    print("DoneCV")

    t1.join()

if __name__ == "__main__":
    main()
