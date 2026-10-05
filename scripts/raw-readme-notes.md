So what is self supervised learning: 
what we are trying to find out is what information can we hide from the model, and what can the model learn to predict from the remaining information. So we have a text, we hide part of the text and we predict what was hidden. 

For example:

Original:
The cat sat on the mat.

Input:
The cat [MASK] on the mat.

Target:
sat

- We are going to build like a small predicition layer. 


We dont want to just train a model here:
we want to understand 
- how self-supervised targets are created
- why predicting missing words is a useful learning objects
- how context allows the model to make predicition
- embeddings
- a basic neural language model
- training with cross-entropy
- how representations improve through self-supervised training
- why this idea eventually leads toward models like BERT and, more broadly, modern language models


Since we did a text based project on our previous unsupervised learning, we can now go to like using images. So like we can hide a certain part of the image and we can like train our model to understand the and comeup with the rest of the image. 

So what is the plan in learning this:
1. Image representation

How do we represent a 28×28 image numerically?

2. Masking

How do we deliberately hide part of an image?

3. Neural network

How can a network take the visible pixels and predict the missing ones?

4. Loss

How do we measure whether its reconstruction is good?

5. Training

Can it improve through gradient descent?

6. Representation

What does the network learn internally while solving this task?

7. Evaluation

Only afterward do we bring back the Fashion-MNIST labels and ask whether the learned representation contains useful information about clothing categories.

That last part is especially important because it lets us demonstrate:

The model was never trained to classify shirts, shoes, bags, etc., but its learned representation may still contain information that distinguishes them.


So in self-supervised learning, the model does not actually need to learn which is a shoe and which is a tshirt, it just learns the structure of pixels, its edges, and it holds them as like general patterns. But since we know the human labels we might know that the stucture of shoes and tshirt have different structure so it like doesnt know explicity know which is which but it knows them by structure. 

And it would think based on the structures i have learned up until now which kind of pixel structure fits here well. 

Then later when we want to check we can compare it with the labeled data and see if the model made actual correct representation. 

- So in building the neural network there is the path it follows:

1. It takes an input of 28 by 28 images, we flatten it to 784 input values, then we push it into our neural networks then we get our 64 output values, which are the 8 by 8 pixels that are missing in the images. 

? Do we need to convert our image into a mathematical format that the model can understand. 

? Later can we try to hid more pixels and see if they work as well, like increase it gradually to see where it starts to fail, by using the same algorithm or maybe a different one

--- Possible research question: how much of an image can we hide before the network can no longer construct the missing structure reliably?


This is the structure or the work the neural network is basically doing:

1. The forward pass answers: Given this masked image, what 64 pixels do I predict?

2. The loss answers: How far are my 64 predictions from the actual 64 pixels?

3. The backward pass answers: How should I change my weights so that my next prediction is better?

