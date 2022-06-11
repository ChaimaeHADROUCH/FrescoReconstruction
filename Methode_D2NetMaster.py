
#############################################################################################
############################### LIBRAIRIES ###############################
#############################################################################################

import argparse
import numpy as np
import imageio
import torch
from tqdm import tqdm
import scipy
import scipy.io
import scipy.misc
from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

## Realiser matching

import cv2
import csv
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
from skimage.feature import match_descriptors

#############################################################################################
############################### PARAMETRES ###############################
#############################################################################################

max_edge = 1600
max_sum_edges = 2800
model_file = 'models/d2_tf.pth'
multiscale = False
output_extension='.d2-net'
output_type='npz'
preprocessing='caffe'
use_relu=True

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Creating CNN model
model = D2Net(
    model_file = model_file,
    use_relu = use_relu,
    use_cuda = use_cuda
)

#############################################################################################
############################### FONCTIONS ###############################
#############################################################################################

def crop_image(img,xC,yC,sizeX,sizeY):
    startx = xC - sizeX//2
    starty = yC - sizeY//2  
    
    img_out = img[starty:(starty+sizeY),startx:(startx+sizeX),:]
    
    while(img_out.shape[0] == 0 or img_out.shape[1]==0):
        sizeX = sizeX - 20        
        sizeY = sizeY - 20
        startx = xC - sizeX//2
        starty = yC - sizeY//2
        img_out = img[starty:(starty+sizeY),startx:(startx+sizeX),:]
    
    return img_out,sizeX,sizeY

def filteringBlackAreaKeypoints(img,keypoints, descriptors):
    new_descriptors = np.ones((1,descriptors.shape[1]))
    new_keypoints = np.ones((1,keypoints.shape[1])) 

    for i in range(len(keypoints)):
        kpt_x = int(keypoints[i][0])
        kpt_y = int(keypoints[i][1])
        img_kpt = img[(kpt_y-2):(kpt_y+3),(kpt_x-2):(kpt_x+3),:]
        
        nbr_zeros=0
        
        for m in range(img_kpt.shape[0]):
            for n in range(img_kpt.shape[1]):
                if(img_kpt[m,n,0]==0):
                    nbr_zeros += 1
                if(img_kpt[m,n,1]==0):
                    nbr_zeros += 1
                if(img_kpt[m,n,2]==0):
                    nbr_zeros += 1
                    
        if (nbr_zeros<3):
            
            new_keypoints = np.concatenate((new_keypoints,[keypoints[i]]),axis=0)
            new_descriptors = np.concatenate((new_descriptors,[descriptors[i]]),axis=0)
            
    return new_keypoints[1:], new_descriptors[1:]

def Detection_keypoints(image,max_edge,max_sum_edges,preprocessing,multiscale):
    
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if max(resized_image.shape) > max_edge:
        resized_image = scipy.misc.imresize(
            resized_image,
            max_edge / max(resized_image.shape)
        ).astype('float')
    if sum(resized_image.shape[: 2]) > max_sum_edges:
        resized_image = scipy.misc.imresize(
            resized_image,
            max_sum_edges / sum(resized_image.shape[: 2])
        ).astype('float')

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(
        resized_image,
        preprocessing=preprocessing
    )
    
    with torch.no_grad():
        if multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales=[1]
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]
    
    return keypoints, descriptors

def get_matrice_transformation(keypoints_fragment, descriptors_fragment,keypoints, descriptors, rotation,L_y, L_x):
        
    matches = match_descriptors(descriptors_fragment, descriptors, cross_check=True)
    
    nbr_matches = matches.shape[0]

    if ( nbr_matches < 3  ):
        print('Error : not enough matches, {} matches'.format(len(mkpts0)))
        return -1,0,0, False
    
    keypoints_left = keypoints_fragment[matches[:, 0], : 2]
    keypoints_right = keypoints[matches[:, 1], : 2]
    
    copy_keypoints_left = np.copy(keypoints_left)
    if (rotation == 90):
        keypoints_left[:, 0] = L_x - copy_keypoints_left[:, 1]
        keypoints_left[:, 1] = copy_keypoints_left[:, 0] 
    elif (rotation == 180):
        keypoints_left[:, 0] = L_x - copy_keypoints_left[:, 0]
        keypoints_left[:, 1] = L_y - copy_keypoints_left[:, 1] 
    elif (rotation == 270):
        keypoints_left[:, 0] = copy_keypoints_left[:, 1]
        keypoints_left[:, 1] = L_y - copy_keypoints_left[:, 0]
    
    transformation_matrix, rigid_mask = cv2.estimateAffinePartial2D(keypoints_left, keypoints_right)
    inliers = np.transpose(rigid_mask[:] == 1 )[0]
    nbr_inliers = np.sum(inliers)
    
    if ( nbr_inliers < 3  ):
        print('Error : not enough matches, {} matches'.format(len(mkpts0)))
        return -1,0,0, False

    bool_accompli = True
    
    return transformation_matrix, nbr_matches , nbr_inliers, bool_accompli
    
def get_transformation(image,image_partie_fresque,max_edge,max_sum_edges,preprocessing,multiscale,sizeX1,sizeY1,Tx,Ty):
#     
    L_y, L_x , L_z  = image.shape
    
    # Detection de Keypoints de la partie de la fresque
    keypoints, descriptors = Detection_keypoints(image_partie_fresque,max_edge,max_sum_edges,preprocessing,multiscale)
    
    startx = Tx - sizeX1//2 
    starty = Ty - sizeY1//2    
    
    keypoints[:,0] += startx
    keypoints[:,1] += starty    
    
    rotation = 0
    keypoints_0, descriptors_0 = Detection_keypoints(image,max_edge,max_sum_edges,preprocessing,multiscale)
    keypoints_0, descriptors_0 = filteringBlackAreaKeypoints(image,keypoints_0, descriptors_0)
    transformation_matrix_0, nbr_matches_0 , nbr_inliers_0,bool_accompli_0 = get_matrice_transformation(keypoints_0, descriptors_0,keypoints, descriptors, rotation,L_y, L_x)
    
    rotation = 90
    image = np.rot90(image)
    keypoints_90, descriptors_90 = Detection_keypoints(image,max_edge,max_sum_edges,preprocessing,multiscale)
    keypoints_90, descriptors_90 = filteringBlackAreaKeypoints(image,keypoints_90, descriptors_90)
    transformation_matrix_90, nbr_matches_90 , nbr_inliers_90,bool_accompli_90 = get_matrice_transformation(keypoints_90, descriptors_90,keypoints, descriptors, rotation,L_y, L_x)
    
    rotation = 180
    image = np.rot90(image)
    keypoints_180, descriptors_180 = Detection_keypoints(image,max_edge,max_sum_edges,preprocessing,multiscale)
    keypoints_180, descriptors_180 = filteringBlackAreaKeypoints(image,keypoints_180, descriptors_180)
    transformation_matrix_180, nbr_matches_180 , nbr_inliers_180,bool_accompli_180 = get_matrice_transformation(keypoints_180, descriptors_180,keypoints, descriptors, rotation,L_y, L_x)
    
    rotation = 270
    image = np.rot90(image)
    keypoints_270, descriptors_270 = Detection_keypoints(image,max_edge,max_sum_edges,preprocessing,multiscale)
    keypoints_270, descriptors_270 = filteringBlackAreaKeypoints(image,keypoints_270, descriptors_270)
    transformation_matrix_270, nbr_matches_270 , nbr_inliers_270,bool_accompli_270 = get_matrice_transformation(keypoints_270, descriptors_270,keypoints, descriptors, rotation,L_y, L_x)
    
    nbr_matches_t = np.array([nbr_matches_0,nbr_matches_90,nbr_matches_180,nbr_matches_270])
    nbr_inliers_t = np.array([nbr_inliers_0,nbr_inliers_90,nbr_inliers_180,nbr_inliers_270])
    
    index_resultat =-1
    
    if(bool_accompli_0 == False and bool_accompli_90 == False and bool_accompli_180 == False and bool_accompli_270 == False ):
        print('Error: Pas possible de trouver Matrice de rotation')
        return 'Nan','Nan','Nan'
    else:
        nbr_inliers_max = np.max(nbr_inliers_t)
        nbr_inliers_max_index = np.where(nbr_inliers_t == nbr_inliers_max)
        
        if(len(nbr_inliers_max_index[0]) > 1 ):
            nbr_matches_t[np.where(nbr_inliers_t != nbr_inliers_max)[0]] = 0
            #print('nbr_matches_t',nbr_matches_t)
            nbr_matches_max = np.max(nbr_matches_t)
            #print('nbr_matches_max',nbr_matches_max)
            nbr_matches_max_index = np.where(nbr_matches_t == nbr_matches_max)
            #print('nbr_matches_max_index',nbr_matches_max_index)            
            index_resultat = nbr_matches_max_index[0][0] 
            #print('nbr_matches_max_index',nbr_matches_max_index)
        else:
            index_resultat = nbr_inliers_max_index[0][0]
            
    if(index_resultat == 0):
#         print('rotation 0', )
        matrice_correct = transformation_matrix_0
    elif(index_resultat == 1):
        matrice_correct = transformation_matrix_90
#         print('rotation 90', )
    elif(index_resultat == 2):
        matrice_correct = transformation_matrix_180
#         print('rotation 180', )
    elif(index_resultat == 3):
        matrice_correct = transformation_matrix_270
#         print('rotation 270', )

    
    T_x = matrice_correct[0,2]
    T_y = matrice_correct[1,2]
    rot = - np.angle(matrice_correct[0,0]+matrice_correct[1,0]*1j)*180/np.pi
    
    if rot > 0 :
        rot=rot-360
    
    tx = -T_y + L_x*np.sin(rot*np.pi/180)/2 - L_y*np.cos(rot*np.pi/180)/2
    ty = -T_x - L_x*np.cos(rot*np.pi/180)/2 - L_y*np.sin(rot*np.pi/180)/2
    
    
    return tx, ty, rot

def Resultat_un_fragmentation(fresques, AllFragmentations, path_AllFragmentations, path_fresques, choix_fresque, choix_fragmentation):

#     count_fragmentation = 0
    
#     for i in range(len(AllFragmentations)):    
#         print('Fragmentation {} : {}'.format(count_fragmentation,AllFragmentations[count_fragmentation]))
#         count_fragmentation += 1

    print('#########################################################################################################')
    print(' ##################### CHOIX FRESQUE :',AllFragmentations[choix_fragmentation],' #####################')
    print('#########################################################################################################')

    path_fragmentation = os.path.join(path_AllFragmentations,AllFragmentations[choix_fragmentation])
    print(path_fragmentation)

    # Process the file
    path_fragments = os.path.join(path_fragmentation,'frag_eroded')
    path_list_fragments = os.path.join(path_fragmentation,'fragments.txt')
    path_imageFresque = os.path.join(path_AllFragmentations,fresques[choix_fresque]+'.ppm')
    path_resultat = os.path.join(path_fragmentation,'resultat_D2NetMaster.csv')

    fichier = open(path_list_fragments, "r")
    text = fichier.read()
    data = text.split('\n')
    image_fresque = np.array(Image.open(path_imageFresque))
    image_fresque = Image.fromarray(image_fresque)
    image_fresque.save(os.path.join(path_AllFragmentations,fresques[choix_fresque]+'.jpg'))
    image_fresque = imageio.imread(os.path.join(path_AllFragmentations,fresques[choix_fresque]+'.jpg'))


    resultat = [['Id','Tx','Ty','Rotation']]

    for i in range(len(data)-1):

        Tx_hat='Nan'
        Ty_hat='Nan'
        rot_hat='Nan'

        id_fragment = int(data[i].split(' ')[0])
        Tx = int(data[i].split(' ')[1])
        Ty = int(data[i].split(' ')[2])
        rot = int(data[i].split(' ')[3])

        print('MATRICE TRANSFORMATION DU FRAGMENT {} :'.format(id_fragment))        
        print('CDV : ',id_fragment,' , ', Tx,' , ', Ty, ' , ', rot)

        # Recuperer les images    
        path = os.path.join(path_fragments,'frag_eroded_{}_color.ppm'.format(id_fragment))
        image = np.array(Image.open(path))
        image = Image.fromarray(image)
        image.save(os.path.join(path_fragments,'frag_eroded_{}_color.jpg'.format(id_fragment)))
        image = imageio.imread(os.path.join(path_fragments,'frag_eroded_{}_color.jpg'.format(id_fragment)))

        sizeX1 = sizeY1 = int(np.mean(image.shape[0:2]))

        image_partie_fresque,sizeX1,sizeY1 = crop_image(image_fresque,Tx,Ty,sizeX1,sizeY1)    

        Tx_hat, Ty_hat, rot_hat = get_transformation(image,image_partie_fresque,max_edge,max_sum_edges,preprocessing,multiscale,sizeX1,sizeY1,Tx,Ty)

        resultat.append([id_fragment,-Ty_hat,-Tx_hat,rot_hat])

        if (Tx_hat != 'Nan' and Ty_hat != 'Nan'and rot_hat != 'Nan'):
            print('Estimation : ',id_fragment,' , ', -Ty_hat,' , ', -Tx_hat, ' , ', rot_hat)
#             print('Translation : (',Tx_hat,',', Ty_hat,')')
#             print('Rotation : ', rot_hat)

    with open(path_resultat, 'w', newline='') as student_file:
        writer = csv.writer(student_file)
        for i in resultat:
            writer.writerow(i)
    
def Resultat_fresque(choix_fresque):
    path_fresques = os.path.join(os.getcwd(),'..','fresques')
    fresques = [f for f in os.listdir(path_fresques) if os.path.isdir(os.path.join(path_fresques, f))]
    
    print('#########################################################################################################')
    print(' ##################### CHOIX FRESQUE :',fresques[choix_fresque],' #####################')
    print('#########################################################################################################')

#     count_fresque = 0

#     for i in range(len(fresques)):    
#         print('Fresque {} : {}'.format(count_fresque,fresques[count_fresque]))
#         count_fresque += 1

    path_AllFragmentations = os.path.join(path_fresques,fresques[choix_fresque])
    AllFragmentations = [f for f in os.listdir(path_AllFragmentations) if os.path.isdir(os.path.join(path_AllFragmentations, f))]
    
    for i in range(len(AllFragmentations)):    
        Resultat_un_fragmentation(fresques, AllFragmentations, path_AllFragmentations, path_fresques, choix_fresque, i)

def Resultat_fresque_fragmentation(choix_fresque, choix_fragmentation):
    path_fresques = os.path.join(os.getcwd(),'..','fresques')
    fresques = [f for f in os.listdir(path_fresques) if os.path.isdir(os.path.join(path_fresques, f))]
    
    print('#########################################################################################################')
    print(' ##################### CHOIX FRESQUE :',fresques[choix_fresque],' #####################')
    print('#########################################################################################################')

#     count_fresque = 0

#     for i in range(len(fresques)):    
#         print('Fresque {} : {}'.format(count_fresque,fresques[count_fresque]))
#         count_fresque += 1

    path_AllFragmentations = os.path.join(path_fresques,fresques[choix_fresque])
    AllFragmentations = [f for f in os.listdir(path_AllFragmentations) if os.path.isdir(os.path.join(path_AllFragmentations, f))]
    
       
    Resultat_un_fragmentation(fresques, AllFragmentations, path_AllFragmentations, path_fresques, choix_fresque, choix_fragmentation)
    
    
#############################################################################################
############################### OBTENIR RESULTATS - CHOISIR FRESQUE ###############################
#############################################################################################

################## CHANGER ICI FRESQUE ##################

choix_fresque = 2

Resultat_fresque(choix_fresque)
