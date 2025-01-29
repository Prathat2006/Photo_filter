# Face Filter Image Processor

A Python-based tool that uses facial recognition to filter and organize images based on face matching. The project offers two variants:
- Single-person face filtering
- Dual-person face filtering

## Features

- Face detection and recognition using InsightFace
- Efficient image processing with OpenCV
- Configurable similarity threshold for match precision
- Support for both single and dual-person face matching
- Processes entire folders of images
- Preserves original images while copying matches to output directory

## Requirements

```
opencv-python
numpy
insightface
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Prathat2006/Photo-filter.git
```

2. Install required packages:
```bash
pip install opencv-python numpy insightface
```

## Usage

### Single Person Face Filtering

```python
# Update the paths in main():
input_folder = 'path/to/input/folder'
output_folder = 'path/to/output/folder'
input_face_image = 'path/to/reference/face.jpg'
```

### Dual Person Face Filtering

```python
# Update the paths in main():
input_folder = 'path/to/input/folder'
output_folder = 'path/to/output/folder'
person1_image = 'path/to/person1/face.jpg'
person2_image = 'path/to/person2/face.jpg'
```

## How It Works

1. The program uses InsightFace to generate face embeddings from reference images
2. It then processes all images in the input folder
3. For single-person mode:
   - Matches images containing the reference face
4. For dual-person mode:
   - Matches images containing both reference faces
5. Matched images are copied to the output folder
6. Similarity threshold can be adjusted (default: 0.6)

## Customization

- Adjust the `threshold` parameter in `process_images()` to control match strictness
- Modify `det_size` in `ImageProcessor` initialization for different detection sizes
- Choose between CPU and GPU providers based on your hardware

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- InsightFace for the face recognition model
- OpenCV for image processing capabilities

## Note

This tool is designed for personal photo organization and should be used responsibly with respect to privacy considerations.