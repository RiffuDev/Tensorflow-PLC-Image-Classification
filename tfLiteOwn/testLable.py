labels_file_path = '/home/rifay/tfLite_imgClass/tfLiteOwn/MMS_sorting_test_tflite_quantized/labels.txt'  # Replace with the actual path to your labels.txt file

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