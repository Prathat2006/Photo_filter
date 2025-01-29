import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

class ImageProcessor:
    def __init__(self, input_folder, output_folder, model_name="buffalo_l", det_size=(640, 640), use_gpu=True):
        self.input_folder = input_folder
        self.output_folder = output_folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Configure providers based on GPU availability
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        try:
            self.app = FaceAnalysis(name=model_name, providers=providers)
            self.app.prepare(ctx_id=0, det_size=det_size)
            print(f"Using providers: {providers}")
        except Exception as e:
            print(f"Error initializing with GPU. Falling back to CPU. Error: {str(e)}")
            self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=det_size)

    
    def get_face_embeddings(self, image_path):
        """Returns all face embeddings found in an image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot read image {image_path}")
            return None
        
        faces = self.app.get(image)
        if len(faces) == 0:
            print(f"No faces detected in {image_path}")
            return None
        
        return [face.normed_embedding for face in faces]
    
    def cosine_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def find_matching_face(self, target_embedding, face_embeddings, threshold):
        """Check if any face in the image matches the target face"""
        if face_embeddings is None:
            return False
            
        for embedding in face_embeddings:
            similarity = self.cosine_similarity(target_embedding, embedding)
            if similarity > threshold:
                return True
        return False
    
    def process_images(self, person1_embedding, person2_embedding, threshold=0.6):
        """Process images looking for both target persons"""
        for filename in os.listdir(self.input_folder):
            input_image_path = os.path.join(self.input_folder, filename)
            
            # Get all face embeddings in the current image
            face_embeddings = self.get_face_embeddings(input_image_path)
            if face_embeddings is None or len(face_embeddings) < 2:
                continue  # Skip if less than 2 faces found
            
            # Check if both target persons are in the image
            person1_found = self.find_matching_face(person1_embedding, face_embeddings, threshold)
            person2_found = self.find_matching_face(person2_embedding, face_embeddings, threshold)
            
            # If both persons are found, copy the image
            if person1_found and person2_found:
                output_image_path = os.path.join(self.output_folder, filename)
                shutil.copy(input_image_path, output_image_path)
                print(f"Match found: Both persons detected in {filename}")

def main():
    input_folder = 'demo_input'
    output_folder = 'demo_output'
    person1_image = "face-1.jpg"   #Add the Path of image  that contains face which you want to filter 
    person2_image = "face-2.jpg"   #Add the Path of image  that contains face which you want to filter 
    
    processor = ImageProcessor(input_folder, output_folder)
    
    # Extract embeddings for both input images
    person1_embedding = processor.get_face_embeddings(person1_image)
    if person1_embedding is None or len(person1_embedding) == 0:
        print("No face found in person1's input image. Exiting.")
        return
    person1_embedding = person1_embedding[0]  # Use first face found
    
    person2_embedding = processor.get_face_embeddings(person2_image)
    if person2_embedding is None or len(person2_embedding) == 0:
        print("No face found in person2's input image. Exiting.")
        return
    person2_embedding = person2_embedding[0]  # Use first face found

    # Process images in the input folder
    processor.process_images(person1_embedding, person2_embedding)

    print(f"Face matching completed! Check '{output_folder}' for images containing both persons.")

if __name__ == "__main__":
    main()