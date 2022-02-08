# HOG-Face-Recognition
A face detection computer vision algorithm that uses Sobel filters, NCC thresholds, and a histogram of oriented gradients to detect faces. Prunes bounding boxes with an intersection of union overlap checking step.

![face_recognition_cropped](https://user-images.githubusercontent.com/77468346/153030770-579c8129-a7e4-402b-93ef-f7577404ec7e.gif)

### extract_hog
The main driver function, performs the image normalization, calls all of the other functions in the
appropriate order, and then calls the visualization function to represent the output.
### get_differential_filter
Generates the sobel filter used for edge detection, for both the x and y directions, allowing the
hog to test for both vertical and horizontal edges.
### filter_image
Applies the given filter (Sobel in this algorithm) to the provided image. It does this by padding
the image to remove edge overlaps and then iterating through all the pixels and multiplying the
pixel intensity values by the filter in a 3x3 radius. For Sobel this emphasizes edges and
suppresses non edge values.
### get_gradient
Generates a gradient descent map represented by a map of the gradient angles and a map of
the gradient magnitudes. It calculates these by iterating through the two filtered images of the x
and y edge values and doing the gradient descent math on each pixel.
### build_histogram
Takes in the generated gradient maps of the angles and magnitudes and categorizes every pixel
into a cell based histogram, it does this by creating a bin map, and assigning every pixel based
on the angle of its gradient, it then adds each histogram into its respective bin and returns the
histogram.
### get_block_descriptor
Normalizes the given histogram to the provided block size, vectorizes the output
### face_detection
Generated a HOG from the template image, then iterated through the pixels of the target image
testing every template-sized section’s HOG for a correlation match to the template HOG using
an NCC threshold of 0.57. It then prunes that list of bounding boxes by removing any that have
an IoU > 0.5 with the highest scoring bounding boxes. Returns the remaining bounding boxes
as the best candidates.

![image](https://user-images.githubusercontent.com/77468346/153023693-bc2798fe-3ffd-43c9-87ae-96150737f7b2.png)
