import cv2
import numpy as np
import os

# folder path
FOLDER_PATH = "ImageSet1"

class ImageClassifier:
    def analyze_and_classify(self, image_path):
        img = cv2.imread(image_path)   
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel = img_hsv[:, :, 0]  
        s_channel = img_hsv[:, :, 1]  
        avg_hue = np.mean(h_channel)
        avg_sat = np.mean(s_channel)
        
        if avg_hue > 5 or avg_sat > 10:
            caption = "day"
        else:
            caption = "night"

        return {
            "filename": os.path.basename(image_path),
            "caption": caption
        }

    def run(self):
        image_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for filename in image_files:
            image_path = os.path.join(FOLDER_PATH, filename)
            result = self.analyze_and_classify(image_path)
            if result:
                img = cv2.imread(image_path)
                window_title = f"Image: {result['filename']} - {result['caption']}"
                cv2.imshow(window_title, img)
                cv2.waitKey(0)  
                cv2.destroyAllWindows()

if __name__ == "__main__":
    classifier = ImageClassifier()
    classifier.run()
