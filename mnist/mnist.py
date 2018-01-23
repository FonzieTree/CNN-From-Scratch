# https://github.com/FonzieTree
import numpy as np
np.random.seed(1)
epsilon = 1e-8
data=np.genfromtxt("train.csv", dtype = 'int', skip_header=1, delimiter = ',')
ytrain = data[:,0]
xtrain = data[:,1:]/255
xtrain = xtrain.reshape(-1, 28, 28)
xtrain[np.argwhere(xtrain==0)] = epsilon
epoch = 200
kernel1 = np.random.randn(64, 3, 3)
kernel2 = np.random.randn(128, 64, 3, 3)
kernel3 = np.random.randn(256, 128, 3, 3)
W = 0.01*np.random.randn(256, 10)
N = xtrain.shape[0]
batch_size = 8
reg = 0.001
lr = 1.0
conv1 = np.zeros((batch_size, 64, 26, 26))
pool1 = np.zeros((batch_size, 64, 13, 13))
relu1 = np.zeros((batch_size, 64, 13, 13))
conv2 = np.zeros((batch_size, 128, 11, 11))
pool2 = np.zeros((batch_size, 128, 5, 5))
relu2 = np.zeros((batch_size, 128, 5, 5))
conv3 = np.zeros((batch_size, 256, 2, 2))
pool3 = np.zeros((batch_size, 256))

dconv3 = np.zeros((batch_size, 256, 2, 2))
dconv2 = np.zeros((batch_size, 128, 11, 11))
dconv1 = np.zeros((batch_size, 64, 26, 26))
dkernel3 = np.zeros((256, 128, 3, 3))
dkernel2 = np.zeros((128, 64, 3, 3))
dkernel1 = np.zeros((64, 3, 3))
drelu2 = np.zeros((batch_size, 128, 5, 5))
drelu1 = np.zeros((batch_size, 64, 13, 13))
for i in range(epoch):
    index = np.array([np.random.randint(0, N) for i in range(batch_size)])
    x = xtrain[index, :, :]
    y = ytrain[index]

    # Feedforward process
    for j in range(batch_size):
        # Layer1
        for k in range(64):
            for o in range(26):
                for p in range(26):
                    conv1[j, k, o, p] = np.sum(x[j, o:o+3, p:p+3]*kernel1[k,:,:])
            for o in range(13):
                for p in range(13):
                    pool1[j, k, o, p] = np.max(conv1[j, k, 2*o:2*o+2, 2*p:2*p + 2])
        relu1[j, :, :, :] = np.maximum(epsilon, pool1[j, :, :, :])

        # Layer2
        for k in range(128):
            for o in range(11):
                for p in range(11):
                    conv2[j, k, o, p] = np.sum(relu1[j, :, o:o+3, p:p+3]*kernel2[k, :, :, :])
            for o in range(5):
                for p in range(5):
                    pool2[j, k, o, p] = np.max(conv2[j, k, 2*o:2*o+2, 2*p:2*p + 2])
        relu2[j, :, :, :] = np.maximum(epsilon, pool2[j, :, :, :])

        # Layer3
        for k in range(256):
            for o in range(2):
                for p in range(2):
                    conv3[j, k, o, p] = np.sum(relu2[j, :, o:o+3, p:p+3]*kernel3[k, :, :, :])
            pool3[j, k] = np.max(conv3[j, k, :, :])
    scores = np.dot(pool3, W)
    exp_scores = np.exp(scores)
    probs = exp_scores/np.sum(exp_scores, axis = 1, keepdims = True)
    correct_logprobs = -np.log(probs[range(batch_size),y])
    data_loss = np.sum(correct_logprobs)/batch_size
    reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(kernel1*kernel1) + 0.5*reg*np.sum(kernel2*kernel2) + 0.5*reg*np.sum(kernel1*kernel1)
    loss = data_loss + reg_loss


    kernel3[np.argwhere(kernel3==0)] = epsilon
    kernel2[np.argwhere(kernel2==0)] = epsilon
    kernel1[np.argwhere(kernel1==0)] = epsilon

    # Backpropagation process
    dscores = probs
    dscores[range(batch_size),y] -= 1
    dscores /= batch_size
    dW = np.dot(pool3.T, dscores)
    dpool3 = np.dot(dscores, W.T)
    for j in range(batch_size):
        # layer3  
        for k in range(256):
            max_locate = np.argwhere(conv3[j, k, :, :] == np.max(conv3[j, k, :, :]))[-1]
            dconv3[j, k, max_locate[0], max_locate[1]] = dpool3[j, k]

            for o in range(2):
                for p in range(2):
                    drelu2[j, :, o:o+3, p:p+3] = dconv3[j, k, o, p]/kernel3[k, :, :, :]
                    dkernel3[k, :, :, :] += dconv3[j, k, o, p]/relu2[j, :, o:o+3, p:p+3]
        dkernel3 = dkernel3/256
        # layer2
        drelu2[j, np.argwhere(pool2[j, :, :, :]<=0)] = 0
        dpool2 = drelu2
        for k in range(128):
            for o in range(5):
                for p in range(5):
                    max_locate = np.argwhere(conv2[j, k, 2*o:2*o+2, 2*p:2*p + 2] == np.max(conv2[j, k, 2*o:2*o+2, 2*p:2*p + 2]))[-1]
                    dconv2[j, k, max_locate[0], max_locate[1]] = dpool2[j, k, o, p]
            for o in range(11):
                for p in range(11):
                    drelu1[j, :, o:o+3, p:p+3] = dconv2[j, k, o, p]/kernel2[k, :, :, :]
                    dkernel2[k, :, :, :] += dconv2[j, k, o, p]/relu1[j, :, o:o+3, p:p+3]
        dkernel2 = dkernel2/128
                # layer1
        drelu1[j, np.argwhere(pool1[j, :, :, :]<=0)] = 0
        dpool1 = drelu1
        for k in range(64):
             for o in range(13):
                 for p in range(13):
                    max_locate = np.argwhere(conv1[j, k, 2*o:2*o+2, 2*p:2*p + 2] == np.max(conv1[j, k, 2*o:2*o+2, 2*p:2*p + 2]))[-1]
                    dconv1[j, k, max_locate[0],max_locate[1]] = dpool1[j, k, o, p]
             for o in range(26):
                 for p in range(26):
                    dkernel1[k, :, :] += dconv1[j, k, o, p]/x[j, o:o+3, p:p+3]
    dkernel3 = dkernel3/(batch_size)
    dkernel2 = dkernel2/(batch_size)
    dkernel1 = dkernel1/(batch_size)
    # Add regularization gradient contribution
    dkernel1 += reg*kernel1
    dkernel2 += reg*kernel2
    dkernel3 += reg*kernel3
    dW += reg*W

    # Update parameters
    kernel1 += -lr*dkernel1
    kernel2 += -lr*dkernel2
    kernel3 += -lr*dkernel3
    W += -lr*dW
    print('Iteraction ', i, ' With loss of ', loss)
