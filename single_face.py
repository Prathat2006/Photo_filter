import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

class ImageProcessor:
    def __init__(self, input_folder, output_folder, model_name="buffalo_l", providers=["CPUExecutionProvider"], det_size=(640, 640)):
        self.input_folder = input_folder
        self.output_folder = output_folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)
    
    def get_face_embedding(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot read image {image_path}")
            return None
        faces = self.app.get(image)
        
        if len(faces) == 0:
            print(f"No face detected in {image_path}")
            return None
        
        return faces[0].normed_embedding  # Return normalized embedding of first face
    
    def cosine_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def process_images(self, user_face_embedding, threshold=0.6):
        for filename in os.listdir(self.input_folder):
            input_image_path = os.path.join(self.input_folder, filename)
            
            face_embedding = self.get_face_embedding(input_image_path)
            if face_embedding is None:
                continue  # Skip if no face found
            
            similarity = self.cosine_similarity(user_face_embedding, face_embedding)
            if similarity > threshold:  # Adjust threshold (higher = more strict)
                output_image_path = os.path.join(self.output_folder, filename)
                shutil.copy(input_image_path, output_image_path)
                print(f"Match found ({similarity:.2f}): Copied {filename}")

# Main function
def main():
    input_folder = '\demo_input'
    output_folder = '\demo_output'
    input_face_image = "face_image" #Add the Path of image  that contains face which you want to filter 
    
    processor = ImageProcessor(input_folder, output_folder)
    
    # Extract embedding for user input image
    user_face_embedding = processor.get_face_embedding(input_face_image)
    if user_face_embedding is None:
        print("No face found in the input image. Exiting.")
        return

    # Process images in the input folder
    processor.process_images(user_face_embedding)

    print(f"Face matching completed! Check '{output_folder}' for matched images.")

if __name__ == "__main__":
    main()