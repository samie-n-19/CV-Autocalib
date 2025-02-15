import cv2
import numpy as np
import os
from scipy.optimize import minimize

image_folder = "Calibration_Imgs"

# Checkerboard parameters
checkerboard_size = (6, 9)
square_size = 21.5 

objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  
imgpoints = []  

image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]

for i, path in enumerate(image_paths):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        
        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        cv2.imwrite(f'detected_corners_{i}.jpg', img)
        
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Function to compute the homography matrix
def compute_homography(objpoints, imgpoints):
    homographies = []
    for i in range(len(imgpoints)):
        H, _ = cv2.findHomography(objpoints[i][:, :2], imgpoints[i], cv2.RANSAC, 5.0)
        homographies.append(H)

    return homographies

# Function to compute V matrix (Eq. 9) as given in the paper
def compute_V(homographies):
    V = []
    for H in homographies:
        h1 = H[:, 0]
        h2 = H[:, 1]
        v12 = [h1[0]*h2[0], h1[0]*h2[1] + h1[1]*h2[0], h1[1]*h2[1],
               h1[2]*h2[0] + h1[0]*h2[2], h1[2]*h2[1] + h1[1]*h2[2], h1[2]*h2[2]]
        v11_minus_v22 = [h1[0]*h1[0] - h2[0]*h2[0], h1[0]*h1[1] + h1[1]*h1[0] - h2[0]*h2[1] - h2[1]*h2[0],
                           h1[1]*h1[1] - h2[1]*h2[1], h1[2]*h1[0] + h1[0]*h1[2] - h2[2]*h2[0] - h2[0]*h2[2],
                           h1[2]*h1[1] + h1[1]*h1[2] - h2[2]*h2[1] - h2[1]*h2[2], h1[2]*h1[2] - h2[2]*h2[2]]
        V.append(v12)
        V.append(v11_minus_v22)
    return np.array(V)

# Function to solve for b (Eq. 9)
def solve_b(V):
    U, S, Vt = np.linalg.svd(V)
    b = Vt[-1]
    b[1] = 0
    return b / b[-1]  

# Function to compute the intrinsic matrix A from b
def compute_A(b):
    B11, B12, B22, B13, B23, B33 = b
    v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lambda_val = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    alpha = np.sqrt(lambda_val / B11)
    beta = np.sqrt(lambda_val * B11 / (B11*B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lambda_val
    u0 = (gamma*v0 / beta) - B13*alpha**2 / lambda_val
    A = np.array([[alpha, gamma, u0],
                  [0, beta, v0],
                  [0, 0, 1]])
    return A

# Function to compute extrinsics (R and t) (Section 3.1)
def compute_extrinsics(A, homographies):
    all_r, all_t = [], []
    for H in homographies:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        lambda_val = 1 / np.linalg.norm(np.linalg.inv(A) @ h1)

        r1 = lambda_val * np.linalg.inv(A) @ h1
        r2 = lambda_val * np.linalg.inv(A) @ h2
        r3 = np.cross(r1, r2)

        t = lambda_val * np.linalg.inv(A) @ h3
        
        R = np.column_stack((r1, r2, r3))

        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        all_r.append(R)
        all_t.append(t)

    return all_r, all_t

def project_points(K, R, t, k1, k2, points):
    points_homogeneous = points.T
    projected = R @ points_homogeneous + t.reshape(3, 1)
    projected = projected.T
    x = projected[:, 0] / projected[:, 2]
    y = projected[:, 1] / projected[:, 2]
    
    r2 = x**2 + y**2
    x_distorted = x * (1 + k1*r2 + k2*r2**2)
    y_distorted = y * (1 + k1*r2 + k2*r2**2)
    
    u = K[0, 0] * x_distorted + K[0, 2]
    v = K[1, 1] * y_distorted + K[1, 2]
    
    return np.column_stack((u, v))


def reprojection_error(params, objpoints, imgpoints):
    fx, fy, cx, cy, k1, k2 = params[:6]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    total_error = 0
    num_points = 0
    for i in range(len(objpoints)):
        R = cv2.Rodrigues(params[6+i*6:6+i*6+3])[0]
        t = params[6+i*6+3:6+i*6+6].reshape(3, 1)
        
        projected = project_points(K, R, t, k1, k2, objpoints[i])
        error = np.sum((imgpoints[i].squeeze() - projected)**2)
        total_error += error
        num_points += len(objpoints[i])
    
    return np.sqrt(total_error / (2 * num_points))  


def callback(xk):
    print(f"Current error: {reprojection_error(xk, objpoints, imgpoints)}")

homographies = compute_homography(objpoints, imgpoints)

V = compute_V(homographies)
b = solve_b(V)

A = compute_A(b)
print("Initial Camera Matrix (A):\n", A)

all_r, all_t = compute_extrinsics(A, homographies)

k1, k2 = 0.0, 0.0
print("Initial distortion parameters:\n", k1, k2)

initial_params = [A[0, 0], A[1, 1], A[0, 2], A[1, 2], k1, k2]
for r, t in zip(all_r, all_t):
    initial_params.extend(cv2.Rodrigues(r)[0].ravel())
    initial_params.extend(t.ravel())

result = minimize(reprojection_error, initial_params, args=(objpoints, imgpoints),
                  method='Powell',
                  options={'maxiter': 1000},
                  callback=callback)

fx, fy, cx, cy, k1_opt, k2_opt = result.x[:6]
K_opt = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

print("Optimized Camera Matrix (K):\n", K_opt)
print("Optimized distortion parameters:\n", k1_opt, k2_opt)

# Save the calibration results asd numpy arrays
np.save("./Camera_params/calibration_matrix", K_opt)
np.save("./Camera_params/distortion_coefficients", np.array([k1_opt, k2_opt]))

final_error = reprojection_error(result.x, objpoints, imgpoints)
print(f"Final reprojection error: {final_error}")

for i in range(len(objpoints)):
    R = cv2.Rodrigues(result.x[6+i*6:6+i*6+3])[0]
    t = result.x[6+i*6+3:6+i*6+6]
    projected = project_points(K_opt, R, t, k1_opt, k2_opt, objpoints[i])
    error = np.sqrt(np.mean(np.sum((imgpoints[i].squeeze() - projected)**2, axis=1)))
    print(f"RMS error for image {i+1}: {error}")

output_folder = "Rectified_Imgs"
os.makedirs(output_folder, exist_ok=True)

for i, path in enumerate(image_paths):
    img = cv2.imread(path)
    h, w = img.shape[:2]

    rectified_img = cv2.undistort(img, K_opt, np.array([k1_opt, k2_opt, 0, 0, 0]), None, A)

    if len(imgpoints) > i and len(objpoints) > i:
        R = cv2.Rodrigues(result.x[6 + i * 6:6 + i * 6 + 3])[0]
        t = result.x[6 + i * 6 + 3:6 + i * 6 + 6]
        projected_corners = project_points(K_opt, R, t, k1_opt, k2_opt, objpoints[i])

        for corner in projected_corners:
            cv2.circle(rectified_img, tuple(int(c) for c in corner), 10, (255, 0, 255), -1)

        reprojection_path = os.path.join(output_folder, f"reprojected_{i}.jpg")
        cv2.imwrite(reprojection_path, rectified_img)

print(f"Rectified and reprojected images saved in {output_folder}.")
