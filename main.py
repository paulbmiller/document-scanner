import cv2
import pytesseract
import numpy as np
from tqdm import tqdm


WIDTH = 640
HEIGHT = 480
FILL_RATIO = 10

pytesseract.pytesseract.tesseract_cmd = \
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def get_contours(img_edges):
    """
    Returns the 4 points which create the biggest contour (which should be the
    corners of the document).

    Parameters
    ----------
    img_edges : numpy.array
        Grayscaled image for edge detection.

    Returns
    -------
    biggest : numpy.array
        Array of the 4 points which create the biggest contour.

    """
    max_area = 0
    biggest = np.array([])
    contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # If the contour fills at least 10 percent of the image
        if area > img_edges.shape[1] * img_edges.shape[0] / FILL_RATIO:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4 and area > max_area:
                biggest = approx
    
    return biggest


def points_to_res(pts, og_res, new_res):
    """
    Change the list of points in another resolution.

    Parameters
    ----------
    pts : np.array
        Array of points.
    og_res : tuple
        Tuple (width, height) of the points.
    og_res : tuple
        Tuple (width, height) of the new resolution we want.

    Returns
    -------
    pts : np.array
        Array of points for a new image resolution.

    """
    new_pts = np.zeros(pts.shape, np.int32)
    width_ratio = new_res[0] / og_res[0]
    height_ratio = new_res[1] / og_res[1]
    
    for i in range(len(pts)):
        new_pts[i] = [pts[i][0] * width_ratio, pts[i][1] * height_ratio]
    
    return new_pts
    

def get_res(img):
    return (img.shape[1], img.shape[0])


def warp(img, biggest):
    """
    Returns the warped image which focuses on the document based on the
    position of the document corners.

    Parameters
    ----------
    img : numpy.array
        Original image.
    biggest : numpy.array
        The position of corners of the document.

    Returns
    -------
    warped_img : numpy.array
        Image of the warped document.

    """
    biggest = biggest.reshape((4, 2))
    biggest = points_to_res(biggest, (WIDTH, HEIGHT), get_res(img))
    coord_sums = np.sum(biggest, axis=1)
    coord_diff = np.diff(biggest, axis=1)
    
    biggest_reordered = np.zeros(biggest.shape, np.int32)
    biggest_reordered[0] = biggest[np.argmin(coord_sums)]
    biggest_reordered[1] = biggest[np.argmin(coord_diff)]
    biggest_reordered[2] = biggest[np.argmax(coord_diff)]
    biggest_reordered[3] = biggest[np.argmax(coord_sums)]    
    
    pts1 = np.float32(biggest_reordered)
    pts2 = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]],
                       [img.shape[1], img.shape[0]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped_img = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
    
    return warped_img


def show_img(img, resolution=(WIDTH, HEIGHT), title='out'):
    """
    Display the given image using OpenCV.

    Parameters
    ----------
    img : numpy.array
        Numpy array of an image.
    resolution : tuple, optional
        Resolution to show. The default is (WIDTH, HEIGHT).
    title : string, optional
        Name of the window. The default is 'out'.

    Returns
    -------
    None.

    """
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, cv2.resize(img, resolution))


def get_text_from_photo(path_to_img):
    """
    Loads the photo from the given path, detects the document and warps the
    image, then removes noise and does OCR with Pytesseract. Returns the text
    which it has recognized.

    Parameters
    ----------
    path_to_img : string
        Path to an image file.

    Returns
    -------
    out_string : string
        Text found in the document from the image.

    """
    og = cv2.imread(path_to_img)
    test = cv2.resize(og, (WIDTH, HEIGHT))
    test = test.copy()
    gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 1)
    canny = cv2.Canny(blur, 150, 150)
    
    biggest = get_contours(canny)
    
    """
    We could also try and find the orientation of the document at this step,
    here we assume that the document is in the right orientation.
    """
    
    if biggest.size != 0:
        warped_img = warp(og, biggest)
        
    else:
        print('Document not found.')
        return ''

    gray_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    dil_warped = cv2.dilate(gray_warped, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dil_warped, 21)
    diff_img = 255 - cv2.absdiff(gray_warped, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    char_wl = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
    conf = '--psm 3 --oem 3 -c tessedit_char_whitelist={}'.format(char_wl)
    out_string = pytesseract.image_to_string(norm_img, config=conf)
    out_string = out_string.replace('\x0c', '')
    
    return out_string


if __name__ == '__main__':
    filenames = [('test_images//1.jpg', 'out//1.txt'),
                 ('test_images//2.jpg', 'out//2.txt'),
                 ('test_images//3.jpg', 'out//3.txt')]
    
    for file in tqdm(filenames, unit='file', desc='Getting text from files'):
        text = get_text_from_photo(file[0])
        
        f = open(file[1], 'w')
        
        f.write(text)
        
        f.close()


