pip3 freeze > requirements.txt
import streamlit as st
from google.colab.patches import cv2_imshow
from IPython.display import clear_output
from PIL import Image
import cv2
import numpy as np
import urllib.request
import tempfile
import base64

weight = '/content/darknet/yolov3.weights'
cfg = '/content/darknet/cfg/yolov3.cfg'

net = cv2.dnn.readNet(weight, cfg)

classes = []
with open("/content/darknet/data/coco.names", "r") as f:
    classes = f.read().splitlines()
print(classes)
def load_image(img):
    im = Image.open(img, 'r')
    return im
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
PAGE_CONFIG = {"page_title":"project.io","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)
@st.cache(suppress_st_warning=True)
def main():
     st.title("OBJECT DETECTION APP")
     st.subheader("CREATED USING GOOGLE COLABORATORY AND STREAMLIT")
     menu = ['IMAGE','VIDEO']
     choice = st.sidebar.selectbox('OPTIONS',menu)
     if choice == 'IMAGE':
         img = st.file_uploader('UPLOAD IMAGE', type = ['jpg', 'jpeg'])
         try:
             img = load_image(img)
         except Exception:
             st.warning('Select an Image')
             st.stop()

         img1 = img
         st.write("Original Image")
         st.image(img1, width =750)
         img1 = np.array(img1)
         height, width, _ = img1.shape
         blob = cv2.dnn.blobFromImage(img1, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
         net.setInput(blob)
         output_layers_names = net.getUnconnectedOutLayersNames()
         layerOutputs = net.forward(output_layers_names)

         boxes = []
         confidences = []
         class_ids = []
         for output in layerOutputs:
             for detection in output:
                 scores = detection[5:]
                 class_id = np.argmax(scores)
                 confidence = scores[class_id]
                 if confidence > 0.2:
                     center_x = int(detection[0]*width)
                     center_y = int(detection[1]*height)
                     w = int(detection[2]*width)
                     h = int(detection[3]*height)
                     x = int(center_x - w/2)
                     y = int(center_y - h/2)
                     boxes.append([x, y, w, h])
                     confidences.append((float(confidence)))
                     class_ids.append(class_id)
         indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
         if len(indexes)>0:
             for i in indexes.flatten():
                 x, y, w, h = boxes[i]
                 label = str(classes[class_ids[i]])
                 confidence = str(round(confidences[i],2))
                 color = colors[i]
                 cv2.rectangle(img1, (x,y), (x+w, y+h), color, 2)
                 cv2.putText(img1, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
         st.write("Final Image")
         st.image(img1, width = 750)
         st.success('FINISHED')
         st.balloons()
         cv2.destroyAllWindows()
     
     else:
         uploaded_file = st.file_uploader('UPLOAD VIDEO', type = ['mp4'])
         try:
             tfile = tempfile.NamedTemporaryFile(delete=False) 
             tfile.write(uploaded_file.read())
         except Exception:
             st.warning('Select a Video')
             st.stop() 
         cap = cv2.VideoCapture(tfile.name)
         stframe = st.empty()
         while True:
             _, img = cap.read()
             img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
             height, width, _ = img.shape
             blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
             net.setInput(blob)
             output_layers_names = net.getUnconnectedOutLayersNames()
             layerOutputs = net.forward(output_layers_names)

             boxes = []
             confidences = []
             class_ids = []
             for output in layerOutputs:
                 for detection in output:
                     scores = detection[5:]
                     class_id = np.argmax(scores)
                     confidence = scores[class_id]
                     if confidence > 0.2:
                         center_x = int(detection[0]*width)
                         center_y = int(detection[1]*height)
                         w = int(detection[2]*width)
                         h = int(detection[3]*height)
                         x = int(center_x - w/2)
                         y = int(center_y - h/2)
                         boxes.append([x, y, w, h])
                         confidences.append((float(confidence)))
                         class_ids.append(class_id)
             indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
             if len(indexes)>0:
                 for i in indexes.flatten():
                     x, y, w, h = boxes[i]
                     label = str(classes[class_ids[i]])
                     confidence = str(round(confidences[i],2))
                     color = colors[i]
                     cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                     cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)        
             stframe.image(img, width = 750)
             cv2.waitKey(0)
             cv2.destroyAllWindows()   
        
if __name__ == '__main__':
     main()
