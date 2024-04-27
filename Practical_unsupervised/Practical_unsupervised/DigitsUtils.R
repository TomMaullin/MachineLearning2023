display_digit = function(img,Lab=NULL,TruLab=NULL,MaxVal=150) {
# Displays digits stored as column vectors in matrix img
#
# img   -  256 x n matrix, where each column corresponds to a 16x16 image
# Lab   -  Optional length n-vector of numbers, to be displayed above the
#          image of the digit.
# TruLab-  Optional length n-vector of numbers, true label; "Lab" is
#          then the predicted label.  If Lab(i) ~= TruLab(i), a message is 
#          shown below the image, indicating the true label.
# MaxVal-  Maximum intensity; by default 150. If set to NA maximum *and*
#          minimum will be set adaptively.
#________________________________________________________________________
# Function: display_digit
# Purpose:  Display grayscale digits with labels
# Author:   T Nichols
# Date:     26 Nov 2019

  nImg=NCOL(img)
  MinVal=0
  # Figure out dimensions of subplots
  if (nImg==1) {
    img=matrix(img,ncol=1)
    N=1
    J=1
  } else {
    N=ceiling(sqrt(nImg)*1.75)
    J=ceiling(nImg/N)
  }
  
  if (is.na(MaxVal)) {
    MinVal=min(img)*.9
    MaxVal=max(img)*.9
  }

  # Loop over images
  par(mfrow=c(J,N),mai = c(0.1, 0.2, 0.1, 0.2),mar=c(0.1,0.2,0.1,0.2))
  for (i in 1:nImg) {
    Title=""
    Xlab=""
    if (!is.null(Lab))
      Title=Lab[i]
    if (!is.null(TruLab) && TruLab[i]!=Lab[i])
      Title=paste0(Title," (",TruLab[i],")")
    image(1:16,1:16,t(matrix(img[,i],nrow=16,ncol=16)[16:1,]),
          col=grey.colors(255),
          zlim=c(MinVal,MaxVal),
          useRaster=TRUE,ylab="",col.axis="red",
          asp=1,xaxt='n',yaxt='n',bty='n')
    title(Title,line=-1)
      }
}
  

train_digit = function(img,lab) {
# FORMAT [ccMean,ccVar] = train_digit(img,lab)
# Fit the class conditional distributions needed to train a Naive
# Bayesian Gaussian classifier
#
# img  - images of handwriting
# lab  - true labels
#
# ccMean - Class conditional means
# ccVar  - Class conditional variances
#
#________________________________________________________________________
# Script:   train_digit
# Purpose:  Train a N.B.C with Gaussian data
# Author:   Tom Nichols
# Date:     10 Jan 2018
#
  
  # Replace zero label with 10
  lab[lab == 0] <- 10
  
  # Make sure lab is a vector
  lab <- c(lab)
  
  # Constants
  nDigit = 10;
  nPixel = NROW(img)

  # sanity check, make sure all digits are present
  if (length(unique(lab)) != 10)
    stop('Not all digits represented!!')

  # Compute class conditional means and variances
  ccMean = ccVar = matrix(0,nrow=nPixel,ncol=nDigit)
  for (i in 1:nDigit) {
    Idx = which(lab==i);
    ccMean[,i] = apply(img[,Idx],1,mean)
    ccVar[,i]  = apply(img[,Idx],1,var)
  }
  
  # Reordering columns so that the last column becomes the first
  ccMean <- ccMean[, c(10, 1:9)]
  ccVar <- ccVar[, c(10, 1:9)]

  list(ccMean=ccMean,ccVar=ccVar)
}



test_digit = function(ccMean,ccVar,img,lab,VarOpt="NoPool") {
# Predict digit identity for images and compare to known labels lab,
# using a Naive Bayesian Classifier
#
# INPUTS
# 
# ccMean - Class conditional means, from training data
# ccVar  - Class conditional variances, from training data
# img    - test data, images of handwriting
# lab    - true labels, to compute accuracy of classification
# VarOpt - Variance option, to control different assumptions on the
#          variance (and deal with 0 variances).  One of
#             "NoPool": Globally constrain variance, so smallest variance is at
#                       least 1/100-th of the maximal variance [Default]
#             "ClassPool": Pool (average) the variance over all classes at each pixel 
#             "AllPool": Pool (average) the variance over all classes, all pixels
#                        (variance becomes a scalar)
#
# OUTPUTS
#
# Accuracy   - Mean number of correctly identified digits
# PredClass  - Predicted class
# 
# Assumes equal prevlance of the 10 digits (P(lab)=1/10))
#
# Regularises the variance, according to VarOpt.
#
#________________________________________________________________________
# Script:   test_digit_classifier
# Purpose:  Test a trained N.B.C with Gaussian data
# Author:   T Nichols
# Date:     26 Nov 2019
#
  
  # Replace zero label with 10
  lab[lab == 0] <- 10
  
  # Make sure lab is a vector
  lab <- c(lab)
  
  # Convert ccMean and ccVar so zero label is 10
  ccMean <- ccMean[, c(2:10,1)]
  ccVar <- ccVar[, c(2:10,1)]

  # Constants
  nDigit = 10;
  nCase  = NCOL(img)

  PredClass = matrix(0,nrow=1,ncol=nCase)

  for (i in 1:nCase) {
    lcc    = logGausClassCond(img[,i],ccMean,ccVar,VarOpt)
    Pred   = which.max(lcc);  # Which element is the max 
    PredClass[i] = Pred
  }

  Accuracy = mean(PredClass==lab)
  
  # Assign back 0
  PredClass[PredClass==10] <- 0

  list(Accu=Accuracy, Pred=PredClass)
}


logGausClassCond = function(x,mn,va,VarOpt) {
# Compute all 10 class conditionals for one image
#
# x  - Column vector of data,   nPixel x 1
# mn - Mean for for each class, nPixel x nDigit
# va - Variance for each class, nPixel x nDigit
# VarOpt - See help for train_digit
  

  nDigit = 10;
  nPixel = NROW(x)
  va_mx  = max(va)

  switch(VarOpt,
    NoPool={
      # Globally constrain the variance, so minimum is no less than 1/100^2 of
      # maximal variance
      va = pmax(va,va_mx/100^2)
    },
    ClassPool={
      # Pool over classes, at each pixel
      va = matrix(apply(va,2,mean),nrow=nPixel,ncol=nDigit)
    },
    AllPool={ 
      # Pool over all classes & pixels
      va = matrix(mean(va),nrow=nPixel,ncol=nDigit)
    },
    {
      stop('Unknown variance option!')
    }
  )

  xx = matrix(x,nrow=nPixel,ncol=nDigit); # Expand data to be nPixel x nDigit, to facilitate
  # computations below (not sure if needed, actually)
  f = -apply(log(va)/2,2,sum) -apply(((xx-mn)^2/va)/2,2,sum)
  f = matrix(f,ncol=nDigit)

  # Terms absent:  -log(2*pi)*P/2 & log(P(Label)), as both are constant for all x
  f

}
 


PCAxform = function(Xtrain,Xtest,K) {
# FORMAT Acc = DimReduction(Xtrain,Xtest,K)
# Using training features, computes PCA transformation and applies to
# both traing and testing features, using a maximum of K dimensions.
#
# INPUTS
#
# Xtrain   - Features for training. 
# Xtest    - Features for testing. 
#
# OUTPUTS
#
# Xs_train - X*, training labels after PCA transformation
# Xs_test  - X*, testing labels after PCA transformation
#
# Dimension of X inputs must be nVariable x nSample
#
# Output Xs are K by nSample, with the variables corresponding to
# PCA directions instead of original input variables.  
#
#________________________________________________________________________
# Script:   PCAxform
# Purpose:  PCA data transformation
# Author:   T Nichols
# Date:     26 Nov 2019
#
  

  nPixel  = NROW(Xtrain)
  n_train = NCOL(Xtrain)
  n_test  = NCOL(Xtest)

  Xtrain_mn = apply(Xtrain,1,mean)

  # Center data, making each pixel mean zero
  Xtrain_C = Xtrain - matrix(Xtrain_mn,nrow=nPixel,ncol=n_train)
  Xtest_C  = Xtest  - matrix(Xtrain_mn,nrow=nPixel,ncol=n_test)

  # Compute full decomp
  tmp    = svd(Xtrain_C)
  Utrain = tmp$u
  S      = tmp$d
  Dtrain = S^2/n_train
  Dtrain = Dtrain/sum(Dtrain)

  Xs_train = t(Utrain[,1:K])%*%Xtrain_C
  Xs_test  = t(Utrain[,1:K])%*%Xtest_C

  Utrain = Utrain[,1:K];
  Dtrain = Dtrain[1:K];

  list(Xs_train=Xs_train,Xs_test=Xs_test,Utrain=Utrain,Dtrain=Dtrain)
}

