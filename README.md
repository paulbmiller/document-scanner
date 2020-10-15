# Document scanner
## Description
This project is for scanning documents using an image in order to warp the perspective with OpenCV.
Then we use optical character recognition to read the information using Pytesseract.

## Input
The images I used for input can be found in the *test_images* folder which are high-resolution images taken from an Android phone in 3000x4000 resolution.

## Output
The ouput text is written to files in the *out* folder.

## Results
It works well for the *1.jpg* and the *2.jpg* images, but does some errors on the *3.jpg* image.
These results are decent, because the 3rd image is taken at an angle and has a big shadow through the text. The text isn't as easy to read even for a human. We could probably get a better angle if we are trying to take a photo of a document.

## Notes
This will only work if the document is in the right orientation and if the image contains all 4 corners of the document clearly.

## Improvements
We could possibly create a document which ressembles the scanned document in LaTeX.