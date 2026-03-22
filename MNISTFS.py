from sklearn.datasets import fetch_openml
import numpy as np
import random

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target
kernel=np.zeros((32,3, 3))

for k1 in range(32): # makes a 32x3x3 matrix of random integers which are kernels
    for k2 in range(3):
        for k3 in range(3): 
            kernel[k1][k2][k3]=random.uniform(-0.1, 0.1)

bias = np.zeros(32)
biasl2=np.zeros(64)
lr=0.0005

W1 = np.random.randn(128,1600) * 0.1
b1 = np.zeros((128,1))

W2 = np.random.randn(10,128) * 0.1
b2 = np.zeros((10,1))

def softmax(Z):           #this is the softmax function which converts everything in probablity terms
    Z_shift = Z - np.max(Z)   # prevent overflow
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ)


kernel_l2=np.zeros((64,32,3,3))
for K1 in range(64):                            #adding  random values to kernels
    for K2 in range(32):
        for K3 in range(3):
            for K4 in range(3):
                kernel_l2[K1][K2][K3][K4]=random.uniform(-0.01, 0.01)
 
def return_max_index(window):                               # this is a specific function designed for the backprop from lower matrix to higher matrix like 64x5x5 to 64x11x11
    return np.unravel_index(np.argmax(window), window.shape)
            
def return_max_number(vec):
    maxval=vec[0][0]
    maxpos=0

    for i in range(len(vec)):
        if vec[i][0]>maxval:
            maxval=vec[i][0]
            maxpos=i
    return maxpos


                
epoch=50
for i in range(epoch):
    totalloss=0
    correct=0
    totall=0
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for j in indices[:500]:
        x=(X[j]/255).reshape(1,28,28)                # takes in 1 image value. and makes the pixel value lie in between 0 n 1.
        convul=np.zeros((32,26,26))                  # Convul holds output of first convulution layer
        for c1 in range(32):
            for c2 in range(26):
                for c3 in range(26):
                    patch=x[0][c2:c2+3,c3:c3+3]                              #patch is the 3x3 part that we will use to apply convulution to
                    convul[c1][c2][c3]=np.sum(patch*kernel[c1])+bias[c1]      #this is actual convulution operation
        output_convul=np.maximum(0,convul)                                   #this is ReLU applied on the convulution layer

                                                                              # convulution for layer 1 is processed
        maxpoooll=np.zeros((32,13,13))                                    
        for m1 in range(32):
            for m2 in range(13):
                for m3 in range(13):
                    window=output_convul[m1][m2*2:(m2+1)*2,m3*2:(m3+1)*2]   #here we take 2x2 matrix and apply stride=2 and get maxpool applied on our output
                    maxpoooll[m1][m2][m3]=np.max(window)                    #this only takes in the strongest signal in the whole matrix.

                                                                            # maxpool layer 1 processed 

        convul_2=np.zeros((64,11,11))               #we use another convulutional layer over here.to gather more features in the image.
        for fil in range(64):
            for height in range(11):
                for width in range(11):
                    total=0
                    for bredth in range(32):
                        patchh=maxpoooll[bredth][height:height+3,width:width+3]
                        total+=np.sum(patchh*kernel_l2[fil][bredth])
                    convul_2[fil][height][width]=total+biasl2[fil]
        output_convul2=np.maximum(0,convul_2)              # this is again relu applied
                                                           #convul layer 2 processed

        maxpool2=np.zeros((64,5,5))                            #maxpool2 initialized
        for M1 in range(64):
            for M2 in range(5):
                for M3 in range(5):
                    windoww=output_convul2[M1][M2*2:(M2+1)*2,M3*2:(M3+1)*2]           #using the 2x2 matrix. stride 2 and reducing the image and identifying stronger signals
                    maxpool2[M1][M2][M3]=np.max(windoww)                         #ReLU

                                                                                #maxpool layer 2 processed
        flat = maxpool2.reshape(-1,1)
                                                                                #did flattening and proceeding to dense layer
        
                                                                                #so this is where dense layer begin.
        Z1 = np.dot(W1, flat) + b1                                          #calculating the first layer Z 1x1600 X 1600x128 = 1x128
        A1 = np.maximum(0, Z1) #ReLU

        Z2 = np.dot(W2, A1) + b2                                                #using a1 as the input for layer 2. 1x128 X 128x10 = 1x10 

        A2 = softmax(Z2)                                                        #applying softmax to 1x10 matrix. which is essentially probablity of being a number
        
        label = int(y[j])
        pred=return_max_number(A2)
        if pred==label:
            correct+=1
        totall+=1

       # calculating loss
        loss = -np.log(A2[label][0] + 1e-9)        #multi class log loss function. L = yi*(log(e^zi/sig(e^zj)))
        totalloss+=loss


        y_values = np.zeros((10,1))                #this is like assigning the highest probablity to the number.
        y_values[label] = 1

        #print("backprop starts")
        DZ2 = A2 - y_values  #DZ2=[d(L)/d(z2)]= -y+A2. pretty intuitive derivative wise.

        relu_grad = (Z1 > 0).astype(int)
        DZ1 = (W2.T @ DZ2) * relu_grad #DZ1=DZ2*[d(Z2)/d(A1)]*[d(A1)/d(Z1)]    where [d(A1)/d(Z1)] is essentially 0 or 1 based on Z2 being greater than or less than 0


#L-Z2-A1-Z1-FLAT-MAXPOOL2-CONVUL2-MAXPOOL1-CONVUL1
#NOW OBJECTIVE IS TO UPDATE THE CNN LAYERS.

        Dfl=W1.T@DZ1                       #fl depends on z1 so d(fl)=d(L)/d(z1)*d(z1)/d(fl). and d(z1)/d(fl) is just W1
        W1 = W1 - lr * np.dot(DZ1, flat.T)            #updating W1 b1 n b2

        b2 -= lr * DZ2
        b1 -= lr * DZ1
        Dmp2=Dfl.reshape(64,5,5)
    

        D_C2=np.zeros((64,11,11))                #derivative of Convulution layer 2. THIS IS WHERE THE REAL TUFF PART COMES.

        for DC1 in range(64):                    #We have to convert a lesser row n column matrix to a higher one.
            for DC2 in range(5):
                for DC3 in range(5):
                    winndow=output_convul2[DC1][DC2*2:(DC2+1)*2,DC3*2:(DC3+1)*2]       # we use the window approach to check the maximum value at the convul2 matrix in front prop and put the DMp2 value which is essentially gradients at that position and assign rest values to 0.
                    max_value=np.max(winndow)
                    r1, r2 = return_max_index(winndow)
                    D_C2[DC1][DC2*2+r1][DC3*2+r2]+=Dmp2[DC1][DC2][DC3]                 # assigning the gradient to the maximum position in 2x2 matrix of convul2

        D_C2=D_C2*(convul_2>0)                                # this is derivative of ReLU which contains basically 0 n 1.                     

        d_maxpool1=np.zeros((32,13,13))                       # initializing the derivative for maxpool1 n kernel 2
        D_Kernel2=np.zeros((64,32,3,3))
        for DMP1 in range(64):
            for DMP2 in range(32):
                for DMP3 in range(11):
                    for DMP4 in range(11):
                        patchhh=maxpoooll[DMP2][DMP3:DMP3+3,DMP4:DMP4+3]                       #patch is a region from 32x13x13 matrix.
                        D_Kernel2[DMP1][DMP2]+=patchhh*D_C2[DMP1][DMP3][DMP4]                       #giving the derivative values of kernel
                        d_maxpool1[DMP2][DMP3:DMP3+3, DMP4:DMP4+3] += (kernel_l2[DMP1][DMP2] * D_C2[DMP1][DMP3][DMP4]) / 64          #THE TUFFEST part of the entire code. dividing by 64 to stablize the network. I dont understand it. 
        D_Kernel2 = np.clip(D_Kernel2, -1, 1)                   # this makes all value in kernel2 in between -1 to 1. just to stablise the whole process and smooth execution
        kernel_l2-=lr*D_Kernel2                                 #updating kernel_l2
        biasl2-=lr*np.sum(D_C2,axis=(1,2))

        d_convul1 = np.zeros((32,26,26))                        #comvul1 initialization

        for DC1 in range(32):
            for DC2 in range(13):
                for DC3 in range(13):
                    window = output_convul[DC1][DC2*2:(DC2+1)*2, DC3*2:(DC3+1)*2]
                    max_val = np.max(window)
                    r1, r2 = return_max_index(window)
                    d_convul1[DC1][DC2*2+r1][DC3*2+r2] += d_maxpool1[DC1][DC2][DC3] # using the same "replacing the max value with slope and rest with 0 approach."

        D_convul1=d_convul1*(convul>0).astype(int)           # ReLU derivative being calculated.

        D_Kernel1=np.zeros((32,3,3))                         #initializing derivative of kernel1

        for K1 in range(32):
            for K2 in range(26):
                for K3 in range(26):
                    patchhhh=x[0][K2:K2+3,K3:K3+3]
                    D_Kernel1[K1]+=patchhhh*D_convul1[K1][K2][K3]                 #computing d_C1
        
        #print("D_Kernel1 sum:", np.sum(np.abs(D_Kernel1)))
        D_Kernel1 = np.clip(D_Kernel1, -1, 1)
        kernel -= lr * D_Kernel1
        bias -= lr*np.sum(D_convul1, axis=(1,2))                                   #updating the final kernels!
        
        

    print("Accuracy:", correct / totall)
    print("Loss:", round(totalloss / totall, 4))                                    #calculating the final accuracy for the test loop.





#although this is decent enough accurate. but it is painstakingly sloww.
        


        
        











