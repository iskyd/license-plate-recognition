import io
import base64
import cv2
import streamlit as st
import numpy as np

st.title('License Plate Recognition')

@st.cache
def text_detect(image):
    textDetector = cv2.dnn_TextDetectionModel_DB("./resources/DB_TD500_resnet50.onnx")
    inputSize = (640, 640)

    # Set threshold for Binary Map creation and polygon detection.
    binThresh = 0.3
    polyThresh = 0.5

    mean = (122.67891434, 116.66876762, 104.00698793)

    textDetector.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh)
    textDetector.setInputParams(1.0/255, inputSize, mean, True)

    boxes, confs = textDetector.detect(image)

    return boxes, confs

@st.cache
def get_vocabulary():
    vocabulary =[]

    # Open file to import the vocabulary.
    with open("./resources/vocabulary.txt") as f:
        # Read the file line by line, and append each into the vocabulary list.
        for l in f:
            vocabulary.append(l.strip())
        f.close()

    return vocabulary

@st.cache
def fourPointsTransform(frame, vertices):
    """Extracts and transforms roi of frame defined by vertices into a rectangle."""
    # Get vertices of each bounding box 
    vertices = np.asarray(vertices).astype(np.float32)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")
    # Apply perspective transform
    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    return result

@st.cache
def text_recognition(image, boxes):
    vocabulary = get_vocabulary()

    # DB model for text-detection based on resnet50.
    textRecognizer = cv2.dnn_TextRecognitionModel("./resources/crnn_cs.onnx")
    textRecognizer.setDecodeType("CTC-greedy")
    textRecognizer.setVocabulary(vocabulary)
    textRecognizer.setInputParams(1/127.5, (100,32), (127.5, 127.5, 127.5), True)

    warped_detection = fourPointsTransform(image, boxes[0])

    textData=[]
    for box in boxes:
        # Apply transformation on the bounding box detected by the text detection algorithm
        croppedRoi  = fourPointsTransform(image,box)
        
        # Recognise the text using the crnn model
        recResult = textRecognizer.recognize(croppedRoi)
        
        # Get scaled values
        boxHeight = int((abs((box[0, 1] - box[1, 1]))))
        
        # Get scale of the font
        fontScale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, boxHeight-30, 1 )
        
        # Write the recognized text on the output image
        placement = (int(box[0, 0]), int(box[0, 1]))

        # Append recognized text to the data storage variable
        textData.append(recResult)

    return ''.join(textData)


uploaded_file = st.file_uploader('Choose an image file:', type=['png','jpg'])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    st.image(img, channels='BGR')

    text_detection_col, text_recognition_col = st.columns(2)

    with text_detection_col:
        st.header('Text Detection')
        # Display processed image.
        output_img = img.copy()
        boxes, confs = text_detect(output_img)

        cv2.polylines(output_img, boxes, True, (255, 0, 255), 4)

        st.image(output_img, channels='BGR')

        with text_recognition_col:
            st.header('Text Recognition')

            text = text_recognition(img, boxes)
            outputCanvas = np.full(img.shape[:3], 255, dtype=np.uint8)
            
            st.text(text)