import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from numpy import extract

def get_differential_filter():
    y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    x = y.T
    return x, y


def filter_image(im, filter):
    padded_orig = np.pad(im, ((1,1),(1,1)), 'constant', constant_values = 0)
    im_filtered = np.zeros(im.shape)
    for x in range(0, len(im)):
        for y in range(0, len(im[0])):
            filter_section = padded_orig[x:x+3,y:y+3]
            im_filtered[x][y] = np.sum(filter_section*filter)
    return im_filtered


def get_gradient(im_dx, im_dy):
    grad_angle = np.zeros(im_dx.shape)
    grad_mag = np.zeros(im_dx.shape)
    for x in range(im_dx.shape[0]):
        for y in range(im_dx.shape[1]):
            grad_mag[x][y] = math.sqrt(math.pow(im_dx[x,y],2) + math.pow(im_dy[x,y],2))
            grad_angle[x][y] = math.atan2(im_dy[x,y],im_dx[x,y])
    grad_angle = grad_angle %  math.pi
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size = 8):
    angles = np.degrees(grad_angle)
    bin_map = np.zeros(angles.shape)
    bin_map = np.zeros(angles.shape)
    for x in range(0, len(angles)):
        for y in range(0, len(angles[0])):
            if(angles[x][y] >= 15 and angles[x][y] < 45): bin_map[x][y] = 1
            elif(angles[x][y] >= 45 and angles[x][y] < 75): bin_map[x][y] = 2
            elif(angles[x][y] >= 75 and angles[x][y] < 105): bin_map[x][y] = 3
            elif(angles[x][y] >= 105 and angles[x][y] < 135): bin_map[x][y] = 4
            elif(angles[x][y] >= 135 and angles[x][y] < 165): bin_map[x][y] = 5
            else: bin_map[x][y] == 0
    ori_histo = np.zeros((int(bin_map.shape[0]/cell_size),int(bin_map.shape[1]/cell_size),6))
    for i in range(0, bin_map.shape[0]):
        for j in range(0, bin_map.shape[1]):
            
            # 3 dimensions of shape of ori_histo
            x = int(i/cell_size)
            y = int(j/cell_size)
            curBin = int(bin_map[i,j])
            if(x == ori_histo.shape[0]): x= (x-1)
            if(y == ori_histo.shape[1]): y= (y-1)
            #throw the each histogram into correspondent angle bins. 
            ori_histo[x,y, curBin] =grad_mag[i,j] + ori_histo[x,y,curBin]
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    Lx = len(ori_histo) - block_size+1
    Ly = len(ori_histo[0]) - block_size+1
    normHisto = np.zeros((Lx, Ly, 6, block_size, block_size))
    for x in range(Lx):
        for y in range(Ly):
            H_i_denom = math.sqrt(np.sum(ori_histo[x:x+block_size, y:y+block_size,:]**2)+(0.001**2))
            normHisto[x,y,:,:,:] = (ori_histo[x:x+block_size, y:y+block_size,:]/(math.sqrt(np.sum(ori_histo[x:x+block_size, y:y+block_size,:]**2)+(0.001**2)))).reshape((6,2,2))
    normHisto = normHisto.reshape((Lx,Ly,6*block_size**2))
    return normHisto


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    normIm = im/np.max(im)
    Fx, Fy = get_differential_filter()
    filteredimx = filter_image(normIm, Fx)
    filteredimy = filter_image(normIm, Fy)
    grad, angle = get_gradient(filteredimx, filteredimy)
    histo = build_histogram(grad, angle, 8)
    hog = get_block_descriptor(histo, 2)
    # visualize to verify
    # visualize_hog(normIm, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()





def face_recognition(I_target, I_template):
    I_target_c= cv2.imread('target.png')
    templateSize = len(I_template)**2
    templateHog = extract_hog(I_template)
    normTempHog = templateHog - np.mean(templateHog)
    bounding = []
    bounding_boxes = []
    nameCt = 0
    for i in range(I_target.shape[0]-I_template.shape[0]):
        for j in range(I_target.shape[1]-I_template.shape[1]):
            testVector = I_target[i:i+(I_template.shape[0]),j:j+(I_template.shape[1])]
            vectorHog = extract_hog(testVector)
            normVecHog = vectorHog - np.mean(vectorHog)
            numerator = np.sum(np.multiply(normTempHog, normVecHog))
            denom = np.linalg.norm(normTempHog) * np.linalg.norm(normVecHog)
            NCC = numerator/denom
            if(NCC > 0.57):
                bounding.append([j, i, NCC])
                if((i+j)%10 == 0):
                    save_face_detection(I_target_c,bounding,str(nameCt),box_size = 50)
                    nameCt +=1
    
    while(len(bounding)>0):
        npBounding = np.array(bounding)
        maxBB = bounding[npBounding[:,2].argmax()]
        bounding_boxes.append(maxBB)
        mxA = maxBB[0]
        myA = maxBB[1]
        mxB = maxBB[0] + len(I_template)
        myB = maxBB[1] + len(I_template[0])
        newBounding = []
        printList = bounding + bounding_boxes
        save_face_detection(I_target_c,printList,str(nameCt),box_size = 50)
        nameCt +=1
        for i in bounding:
            xA = max(mxA, i[0])
            yA = max(myA, i[1])
            xB = min(mxB, (i[0] + len(I_template)))
            yB = min(myB, (i[1] + len(I_template[0])))
            overlap = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            IoU = overlap/(templateSize+templateSize-overlap)
            # if((i)%10 == 0):
            #     save_face_detection(I_target,bounding,str(nameCt),box_size = 50)
            #     nameCt +=1
            if(IoU < 0.5):
                newBounding.append(i)
        bounding = newBounding
    
    bounding_boxes = np.array(bounding_boxes)
    return  bounding_boxes


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()

def save_face_detection(I_target,bounding_boxes,Fname,box_size = 50):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.savefig("images/" + Fname)


if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)

    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    #this is visualization code.



