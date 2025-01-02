import cv2 as cv


'''The next step consists in content indexing. For each image in the test set you should:
• Extract the SIFT descriptors of the feature points in the image,
• Project the descriptors onto the codebook, i.e., for each descriptor the
closest cluster prototype should be found,
• Construct the generated corresponding bag of words, i.e, word histogram.
Please note that you have already performed the same steps for the training images during codebook generation.
Now construct and save a table that would contain, per entry at least the file name, the true category, if it 
belongs to the training- or test set, and the corresponding bag of words / word histogram. The table need only be
 computed once and then used repeatably in the following retrieval experiments.'''