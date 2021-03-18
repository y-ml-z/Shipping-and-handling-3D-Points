import matplotlib.pyplot as plt
import numpy as np
import cv2
import math as m
from numpy.linalg import svd
#1/a/ 
cordonnees_2D_droite = np.array([[1023,600],[1004,719],[984,959],[834,419],[700,442],[580,458],[474,473],[1003,221],[972,135],[945,58],[1204,396],[1278,408],[1341,418],[1397,542]])
cordonnees_3D = np.array([[0, -7.6, 26.281],[0, -12.7, 17.298],[0, -17.7, 8.490],[7.5, 0, 26.457],[12.6, 0, 17.474],[17.6, 0, 8.666],[22.2, 0, 0.000],[0, 7.5, 26.457 ],[0, 12.5, 17.650],[0, 17.6, 8.666],[-7.8, 0, 25.929],[-12.65, 0, 17.386],[-17.65, 0, 8.578],[-22.65, 0, 0.000]])
K_droite = np.zeros((2*cordonnees_2D_droite.shape[0],11))
for k in range(0,K_droite.shape[0],2):
    K_droite[k,0:3] = cordonnees_3D[round(k/2),:]
    K_droite[k,3] = 1
    K_droite[k,4:8] = 0
    K_droite[k,8:] = -cordonnees_2D_droite[round(k/2),0]*cordonnees_3D[round(k/2),:]
    K_droite[k+1,0:4] = 0
    K_droite[k+1,4:7] = cordonnees_3D[round(k/2),:]
    K_droite[k+1,7] = 1
    K_droite[k+1,8:] = -cordonnees_2D_droite[round(k/2),1]*cordonnees_3D[round(k/2),:]  
C_droite = np.zeros((2*cordonnees_2D_droite.shape[0],1))
for j in range(2*cordonnees_2D_droite.shape[0]):
    C_droite[j] = cordonnees_2D_droite[j//2,j%2]
A = np.dot(K_droite.T,K_droite)
B = np.linalg.inv(A)
P = np.dot(B,K_droite.T)
Q_droite = np.dot(P,C_droite)
m34_d = 1/m.sqrt(Q_droite[8,0]**2 + Q_droite[9,0]**2 + Q_droite[10,0]**2) 
M_droite = m34_d*np.array([[Q_droite[0,0], Q_droite[1,0], Q_droite[2,0],Q_droite[3,0]],     
                           [Q_droite[4,0], Q_droite[5,0], Q_droite[6,0],Q_droite[7,0]],
                           [Q_droite[8,0], Q_droite[9,0], Q_droite[10,0],1]])
cordonnees_2D_gauche = np.array([[1206,730],[1161,852],[1118,959],[1013,563],[847,594],[698,619],[564,641],[1159,356],[1087,273],[1025,199],[1362,517],[1398,526],[1427,533],[1453,542]])
cordonnees_3D = np.array([[0, -7.6, 26.281],[0, -12.7, 17.298],[0, -17.7, 8.490],[7.5, 0, 26.457],[12.6, 0, 17.474],[17.6, 0, 8.666],[22.2, 0, 0.000],[0, 7.5, 26.457 ],[0, 12.5, 17.650],[0, 17.6, 8.666],[-7.8, 0, 25.929],[-12.65, 0, 17.386],[-17.65, 0, 8.578],[-22.65, 0, 0.000]])
K_gauche = np.zeros((2*cordonnees_2D_gauche.shape[0],11))
for i in range(0,K_gauche.shape[0],2):
    K_gauche[i,0:3] = cordonnees_3D[round(i/2),:]
    K_gauche[i,3] = 1
    K_gauche[i,4:8] = 0
    K_gauche[i,8:] = -cordonnees_2D_gauche[round(i/2),0]*cordonnees_3D[round(i/2),:]
    K_gauche[i+1,0:4] = 0
    K_gauche[i+1,4:7] = cordonnees_3D[round(i/2),:]
    K_gauche[i+1,7] = 1
    K_gauche[i+1,8:] = -cordonnees_2D_gauche[round(i/2),1]*cordonnees_3D[round(i/2),:]    
C_gauche = np.zeros((2*cordonnees_2D_gauche.shape[0],1))
for j in range(2*cordonnees_2D_gauche.shape[0]):
    C_gauche[j] = cordonnees_2D_gauche[j//2,j%2]
A = np.dot(K_gauche.T,K_gauche)
B = np.linalg.inv(A)
P = np.dot(B,K_gauche.T)
Q_gauche = np.dot(P,C_gauche)
m34_g = 1/m.sqrt(Q_gauche[8,0]**2 + Q_gauche[9,0]**2 + Q_gauche[10,0]**2) 
M_gauche = m34_g * np.array([[Q_gauche[0,0], Q_gauche[1,0], Q_gauche[2,0],Q_gauche[3,0]],     
                           [Q_gauche[4,0], Q_gauche[5,0], Q_gauche[6,0],Q_gauche[7,0]],
                           [Q_gauche[8,0], Q_gauche[9,0], Q_gauche[10,0],1]])  
print('the right camera projection matrix is =', M_droite)                             
print('the left camera projection matrix is =', M_gauche)
#b/
print("the height of the right camera is = :",m34_d)
print("the height of the left camera is = :",m34_g)
cordonnees_2D_droite_cal = np.mat([0,0,0]).T
g = np.mat([1,1,1,1]).T
g[0:3] = np.mat([cordonnees_3D[0]]).T
cordonnees_2D_droite_cal = np.dot(M_droite,g)

for i in range(1,cordonnees_3D.shape[0]):
    g[0:3] = np.mat([cordonnees_3D[i]]).T
    cordonnees_2D_droite_cal = np.concatenate((cordonnees_2D_droite_cal,np.dot(M_droite,g)),axis=1)

for j in range(cordonnees_2D_droite_cal.shape[1]):
    cordonnees_2D_droite_cal[:,j] = cordonnees_2D_droite_cal[:,j]/cordonnees_2D_droite_cal[2,j]
e_d = 0
for i in range(cordonnees_2D_droite_cal.shape[1]):
    e_d = e_d + m.sqrt((cordonnees_2D_droite[i,0] - cordonnees_2D_droite_cal[0,i])**2 + (cordonnees_2D_droite[i,1] - cordonnees_2D_droite_cal[1,i])**2)
e_d = e_d / cordonnees_2D_droite_cal.shape[1]
cordonnees_2D_gauche_cal = np.mat([0,0,0]).T
g = np.mat([1,1,1,1]).T
g[0:3] = np.mat([cordonnees_3D[0]]).T
cordonnees_2D_gauche_cal = np.dot(M_gauche,g)
for i in range(1,cordonnees_3D.shape[0]):
    g[0:3] = np.mat([cordonnees_3D[i]]).T
    cordonnees_2D_gauche_cal = np.concatenate((cordonnees_2D_gauche_cal,np.dot(M_gauche,g)),axis=1)
for j in range(cordonnees_2D_gauche_cal.shape[1]):
    cordonnees_2D_gauche_cal[:,j] = cordonnees_2D_gauche_cal[:,j]/cordonnees_2D_gauche_cal[2,j]
e_g = 0
for i in range(cordonnees_2D_gauche_cal.shape[1]):
    e_g = e_g + m.sqrt((cordonnees_2D_gauche[i,0] - cordonnees_2D_gauche_cal[0,i])**2 + (cordonnees_2D_gauche[i,1] - cordonnees_2D_gauche_cal[1,i])**2)  
e_g = e_g / cordonnees_2D_gauche_cal.shape[1]
print('the error of the right cam (Rd) =  ',e_d)
print('the left camera error (Rg) =  ',e_g)
#c/
F, status = cv2.findFundamentalMat(cordonnees_2D_droite[:9,:], cordonnees_2D_gauche[:9,:])
E, statuts1 = cv2.findEssentialMat(cordonnees_2D_gauche,cordonnees_2D_droite)
print('the essential matrix =', E)
print('the fundamental matrix =', F)
#d/ The optical center has for coordinates: (X, Y, Z) = (0,0,0) in the frame of the camera
#2.a/
K = np.ones((8,9))
for i in range(K.shape[0]):
    K[i,:] = [cordonnees_2D_gauche[i,0]*cordonnees_2D_droite[i,0],cordonnees_2D_gauche[i,1]*cordonnees_2D_droite[i,0],cordonnees_2D_droite[i,0,],cordonnees_2D_gauche[i,0]*cordonnees_2D_droite[i,1],cordonnees_2D_gauche[i,1]*cordonnees_2D_droite[i,1],cordonnees_2D_droite[i,1],cordonnees_2D_gauche[i,0],cordonnees_2D_gauche[i,1],1]
def  nullspace ( A ,  atol = 1e-13 ,  rtol = 0 ): 
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns
F_meth_lin = np.array([[nullspace(K)[0,0], nullspace(K)[1,0], nullspace(K)[2,0]],
                       [nullspace(K)[3,0], nullspace(K)[4,0], nullspace(K)[5,0]],
                       [nullspace(K)[6,0], nullspace(K)[7,0], nullspace(K)[8,0]]])
print('the fundamental matrix by a linear method Fm =', F_meth_lin)
#b/    
pixel_gauche = np.array([[657],[729],[1]])
L_D = np.dot(F_meth_lin.T,pixel_gauche)
image_droite = cv2.imread('/home/lenovo/Desktop/droiter.png',0)
x = np.linspace(1,image_droite.shape[1],image_droite.shape[1])
y = -(L_D[0,0]/L_D[1,0])*x - L_D[2,0]/L_D[1,0]
plt.plot(x,y)
plt.imshow(image_droite)
def droite_epipolaire(pix_g,fond_mat,image_drte):
    p_g = np.array([[pix_g[0]],[pix_g[1]],[1]])
    coef_drt = np.dot(fond_mat.T,p_g)
    u = np.linspace(1,image_drte.shape[1],image_drte.shape[1])
    v = -(coef_drt[0,0]/coef_drt[1,0])*u - coef_drt[2,0]/coef_drt[1,0]
    return plt.plot(u,v),plt.imshow(image_drte)
#c/
print('the coordinates of the right and left epipole = ',(nullspace(F)[0,0],nullspace(F)[1,0]),(nullspace(F.T)[0,0],nullspace(F.T)[1,0]))
ffd = cv2.FastFeatureDetector_create(threshold=25) 
kp_d = ffd.detect(image_droite,None)
img_d_pt_inter = cv2.drawKeypoints(image_droite, kp_d, None, color=(255,0,0))
img_gauche = cv2.imread('/home/lenovo/Desktop/gaucher.png',0)
ffd = cv2.FastFeatureDetector_create(threshold=25)
kp_g = ffd.detect(img_gauche,None)
img_g_pt_inter = cv2.drawKeypoints(img_gauche, kp_g, None, color=(255,0,0))
#d/
im1 = cv2.imread('/home/lenovo/Desktop/gaucher.png',0)  
im2 = cv2.imread('/home/lenovo/Desktop/droiter.png',0) 
sift = cv2.xfeatures2d.SIFT_create()
keyp_1, desc_1 = sift.detectAndCompute(im1,None)
keyp_2, desc_2 = sift.detectAndCompute(im2,None)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(desc_1,desc_2)
matches = sorted(matches, key = lambda x:x.distance)
im = cv2.drawMatches(im1, keyp_1, im2, keyp_2, matches[:50], im2, flags=2)
plt.imshow(im),plt.show()
#3/
im_g = cv2.imread('/home/lenovo/Desktop/gaucher.png',0)
im_d = cv2.imread('/home/lenovo/Desktop/droiter.png',0)
IM = cv2.StereoBM_create(numDisparities=16, blockSize=15)
im = IM.compute(im_g,im_d)
plt.imshow(im,'gray')
plt.show()
