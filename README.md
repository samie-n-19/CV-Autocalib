# Camera Calibration

## Overview
This project implements camera calibration using a checkerboard pattern to estimate intrinsic and extrinsic parameters. The calibration process involves detecting checkerboard corners, computing homographies, solving for the camera's intrinsic matrix, and optimizing parameters through non-linear minimization. The final results include an optimized camera matrix, distortion coefficients, and rectified images.

## Requirements
To run this project, you need the following dependencies:
- Python 3.x
- OpenCV
- NumPy
- SciPy

Install the required packages using:
```bash
pip install opencv-python numpy scipy
```

## Usage
### 1. Place Calibration Images
Store all checkerboard images in a folder named `Calibration_Imgs`.

### 2. Run Calibration Script
Execute the following command:
```bash
python3 Wrapper.py
```

### 3. Outputs
- Detected corners images: `detected_corners_X.jpg`
- Optimized camera matrix and distortion coefficients saved in `Camera_params/`
- Rectified images stored in `Rectified_Imgs/`

<img width="1318" height="1649" alt="image" src="https://github.com/user-attachments/assets/dd5203f2-7550-4ae1-b702-028d203facfd" />


