import argparse
import numpy as np
import cv2 as cv
import pickle
from scipy.spatial.transform import Rotation as R

# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using chessboard images.')
parser.add_argument('file', type=str, help='Name of the file containing data (*pkl).')
parser.add_argument( '-e', '--extended', action='store_true', default=False, help='Enables use of cv.calibrateCameraExtended instead of the default function.')
parser.add_argument( '-s', '--save', action='store_true', default=False, help='Saves calibration data results in .yml format.')
parser.add_argument('-fd', '--filterdist', action='store_true', default=False, help='Enables filter by distance of camera position.')
parser.add_argument('-ft', '--filtertime', action='store_true', default=False, help='Enables filter by time between frames.')
parser.add_argument('-md', '--mindist', type=float, metavar='N', default=0.0, help='Minimum distance between cameras (available only when --filterdist is active).')
parser.add_argument('-rd', '--reduction', type=int, metavar='N', default=1, help='Reduction of number of frames (total/N) (available only when --filtertime is active).')
parser.add_argument('-rs', '--residue', type=int, metavar='N', default=0, help='Residue or offset for the reduced number of frames (available only when --filtertime is active).')

###############################################################################################################################
# Functions

def split_by_distance(objpts, imgpts, names, vecs, min_dist):
    # Get distance of the camera between frames using rvec and tvec and return the lists of frames with a difference over "min_dist".
    arg_split = []
    for i in range(len(vecs)):
        rvec = vecs[i][0]
        tvec = vecs[i][1]
        rmat = R.from_rotvec(rvec.reshape(3))
        tmat = np.dot(rmat.as_matrix(), tvec)
        if i == 0:
            tmat_old = tmat
            arg_split.append(i)
        else:
            dtc = np.linalg.norm(tmat_old - tmat)
            if dtc >= min_dist:
                tmat_old = tmat
                arg_split.append(i)
    
    # After getting all frames with a significant distance, filter the 3D, 2D and name lists to have only them.  
    nobj = [objpts[i] for i in range(len(objpts)) if i in arg_split]
    nimg = [imgpts[i] for i in range(len(imgpts)) if i in arg_split]
    nnames = [names[i] for i in range(len(names)) if i in arg_split]
    return nobj, nimg, nnames

###############################################################################################################################
# Flags

# CALIB_USE_INTRINSIC_GUESS: Calibration needs a preliminar camera_matrix to start (necessary in non-planar cases)
flags_model = cv.CALIB_USE_INTRINSIC_GUESS

###############################################################################################################################
# Main

# Get parser arguments
args = parser.parse_args()

# Load pickle file
print(f'Loading {args.file}.pkl')
pFile = pickle.load(open(f"./datasets/pkl-files/{args.file}.pkl","rb"))

# Unpack lists
objpoints = pFile['3D_points']
imgpoints = pFile['2D_points']
ret_names = pFile['name_points']

camera_matrix = pFile['init_mtx']
dist_coeff = pFile['init_dist']
img_shape = pFile['img_shape']

calibfile = pFile['init_calibfile']
vecs = pFile['rt_vectors']

# Filter lists if required
if args.filterdist:
    print(f'Filter by distance enabled')
    objpoints, imgpoints, ret_names = split_by_distance(objpoints, imgpoints, ret_names, vecs, args.mindist)
    
elif args.filtertime:
    print(f'Filter by time enabled')
    objpoints = objpoints[args.residue::args.reduction]
    imgpoints = imgpoints[args.residue::args.reduction]
    ret_names = ret_names[args.residue::args.reduction]
    
print(f'Length of lists for calibration: {len(ret_names)}')

# Camera Calibration
print("Calculating camera parameters...")
if args.extended:
    ret, mtx, dist, rvecs, tvecs, stdInt, stdExt, pVE = cv.calibrateCameraExtended(objpoints, imgpoints, img_shape, cameraMatrix=camera_matrix, distCoeffs=dist_coeff, flags=flags_model)
    pVE_extended = np.array((np.array(ret_names, dtype=object), pVE[:,0])).T
    pVE_extended = pVE_extended[pVE_extended[:,1].argsort()]
else:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_shape, cameraMatrix=camera_matrix, distCoeffs=dist_coeff, flags=flags_model)

print('Camera matrix:\n', mtx)
print('Distortion coefficients:\n', dist)
if args.extended:
    print('Error per frame:\n', pVE_extended)

if args.save:
    summary = input("Insert comments: ")
    fs = cv.FileStorage('./results/'+args.file[:-14]+'.yml', cv.FILE_STORAGE_WRITE)
    fs.write('summary', summary)
    fs.write('init_cam_calib', calibfile) # ???????????????????
    fs.write('camera_matrix', mtx)
    fs.write('dist_coeff', dist)
    if args.extended:
        pVElist = np.array((np.array([int(x[5:]) for x in ret_names]), pVE[:,0])).T
        fs.write('std_intrinsics', stdInt)
        fs.write('std_extrinsics', stdExt)
        fs.write('per_view_errors', pVElist)
    fs.release()

print("We finished!")