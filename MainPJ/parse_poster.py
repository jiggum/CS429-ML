import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from PIL import Image
import pytesseract


SAME_BOX_THRESHORD = 5

# num of inner boxs
# 1 / num of inner boxs
# average size of inner boxs
# mid of inner boxs
# variance of inner boxs
# size of outer boxs (w/h each)
# 1/size of outer boxs (w/h each)
# num of integers
# num of alpabets

class Figure:
    def __init__(self,shape, rects, text):
        areas = [rect.area for rect in rects]
        self.num_box = len(rects)  # num of inner boxs
        if self.num_box==0:
            self.num_box = 1;
            areas = [0]
        self.num_box_inv = 1.0/self.num_box  # 1 / num of inner boxs
        self.avg_box = np.average(areas)  # average size of inner boxs
        self.mid_box = np.median(areas)  # middian of inner boxs
        self.var_box = np.var(areas)  # variance of inner boxs
        self.width = shape[1]
        self.width_inv = 1.0/shape[1]
        self.heigth = shape[0]
        self.heigth_inv = 1.0/shape[0]
        self.num_int = sum(c.isdigit() for c in text)  # num of integers
        self.num_alpha = sum(c.isalpha() for c in text)  # num of alpabets

class Rect:
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w*h
    def get_tuple(self):
        return ((self.x,self.y),(self.w,self.h),0)
    def change_size(self, ratio, bias=0):
        self.w = int(self.w * ratio + bias)
        self.h = int(self.h * ratio + bias)
    def change_location(self, ratio, bias=0):
        self.x = int(self.x * ratio + bias)
        self.y = int(self.y * ratio + bias)

def dectect_boxs_in_block(img_origin): #
    img_w, img_h = img_origin.shape
    img_gray = cv2.cvtColor(img_origin, cv2.COLOR_GRAY2BGR)
    #img_after_canny = cv2.Canny(img_origin, 100, 300, 3)
    #element_morp = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
    thresh, bw_img = cv2.threshold(img_origin, 255, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    element_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1));
    img_after_erode = cv2.erode(bw_img, element_erode)
    #img_after_morp = cv2.morphologyEx(img_after_erode, cv2.MORPH_CLOSE, element_morp)
    image, contours, hierarchy = cv2.findContours(img_after_erode.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour in contours: #remove simmilar boxs
        #rect = cv2.minAreaRect(contours[i])
        x, y, w, h = cv2.boundingRect(contour)
        rect = Rect(x+w/2,y+h/2,w,h)
        input_check = True
        """
        for rect_target in rects: # remove simmilar box (it's very high cost)
            if(np.linalg.norm([rect.x-rect_target.x,rect.y-rect_target.y])< SAME_BOX_THRESHORD and \
               np.linalg.norm([(rect.x+rect.w)-(rect_target.x+rect_target.w),(rect.y+rect.h)-(rect_target.y+rect_target.h)])< SAME_BOX_THRESHORD):
                input_check = False
                break
            if(np.linalg.norm([(rect.x+rect.w)-(rect_target.x+rect_target.w),rect.y-rect_target.y])< SAME_BOX_THRESHORD and \
               np.linalg.norm([rect.x-rect_target.x,(rect.y+rect.h)-(rect_target.y+rect_target.h)])< SAME_BOX_THRESHORD):
                input_check = False
                break
        """
        if input_check and rect.area>30 and rect.area< img_w*img_h/5:
            rects.append(rect)

    ##print block
    """
    for rect in rects: #draw boxs to image
        box = cv2.boxPoints(rect.get_tuple())
        box = np.int0(box)
        img_gray = cv2.drawContours(img_gray,[box],0,(random.randrange(0,256),random.randrange(0,256),random.randrange(0,256)),1)

    while(2):
        cv2.imshow("cropped", img_gray)
        if cv2.waitKey(33) == 27:
            break

    show_imgs([img_origin, bw_img, img_after_erode, img_gray])
    """

    return rects

def get_figures_of_block(img, rects_inner): # use dectect_boxs_in_block, automatcally make figure class
    shape = img.shape
    img_ocr = Image.fromarray(img)
    txt = pytesseract.image_to_string(img_ocr, lang='Eng')
    """
    print "text :" ,txt
    while(2):
        cv2.imshow("cropped", img)
        if cv2.waitKey(33) == 27:
            break
    """
    return Figure(shape, rects_inner, txt)

def get_figures_of_blocks(img, rects_outter):
    figures = []
    for rect in rects_outter:
        rect.change_size(1.0/resize_ratio,padding)
        rect.change_location(1.0/resize_ratio,padding)
        crop_img = cv2.getRectSubPix(img_origin, rect.get_tuple()[1], rect.get_tuple()[0])
        rects_inner = dectect_boxs_in_block(crop_img)
        figure = get_figures_of_block(crop_img, rects_inner)
        figures.append(figure)
    return figures

def print_figures(figures):
    for figure in figures:
        attrs = vars(figure)
        print ', '.join("%s %s, " % item for item in attrs.items())
def show_imgs(img_array):
    n = 100+len(img_array)*10 + 1
    for i, img in enumerate(img_array):
        plt.xticks([]), plt.yticks([])
        plt.subplot(n+i), plt.imshow(img, cmap='gray')
    plt.show()

def detect_letter_boxs_with_size(img_origin, ratio):
    rows, cols = img_origin.shape
    img = cv2.resize(img_origin,(int(cols*ratio),int(rows*ratio)))
    canny = cv2.Canny(img, 200, 300)
    #sobel = cv2.Sobel(img_origin, cv2.CV_8U, dx=1, dy=0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    #ret, sobel1 = cv2.threshold(sobel,127,255,0)
    #ret, sobel = cv2.threshold(sobel,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    img_after_edge = canny  # selct sobel or canny filter
    element_morp = cv2.getStructuringElement(cv2.MORPH_RECT, (13,2))
    element_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4));
    img_after_morp = cv2.morphologyEx(img_after_edge, cv2.MORPH_CLOSE, element_morp)
    img_after_erode = cv2.erode(img_after_morp, element_erode)
    image, contours, hierarchy = cv2.findContours(img_after_erode.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for i in range(len(contours)):
        #rect = cv2.minAreaRect(contours[i])
        x, y, w, h = cv2.boundingRect(contours[i])
        rect = Rect(x+w/2,y+h/2,w,h)
        rect.change_location(1.0/ratio)
        rect.change_size(1.0/ratio, 6/ratio)
        if(rect.w>20 and rect.w > rect.h):
            rects.append(rect)
    return img_after_erode, rects

def remove_duplicated_rects(rects_target, rects_input):
    for rect_input in rects_input:
        input_check = True
        for rect_target in rects_target:
            if(np.linalg.norm([rect_input.x-rect_target.x,rect_input.y-rect_target.y])< SAME_BOX_THRESHORD and \
               np.linalg.norm([(rect_input.x+rect_input.w)-(rect_target.x+rect_target.w),(rect_input.y+rect_input.h)-(rect_target.y+rect_target.h)])< SAME_BOX_THRESHORD):
                input_check = False
                break
            if(np.linalg.norm([(rect_input.x+rect_input.w)-(rect_target.x+rect_target.w),rect_input.y-rect_target.y])< SAME_BOX_THRESHORD and \
               np.linalg.norm([rect_input.x-rect_target.x,(rect_input.y+rect_input.h)-(rect_target.y+rect_target.h)])< SAME_BOX_THRESHORD):
                input_check = False
                break
        if input_check:
            rects_target.append(rect_input)
    return rects_target

def detect_letter_boxs(img_origin):
    # remove edge padding of the poster
    rows_origin, cols_origin = img_origin.shape
    padding = rows_origin/100
    img = img_origin[padding:rows_origin-padding, padding:cols_origin-padding]
    rows, cols = img.shape

    """
    if(rows > 2000):
        img_resized = cv2.resize(img,(int(2000.0/rows*cols),2000))
        rows_resized, cols_resized = img.shape
    """
    resize_ratio = 300.0/rows
    img = cv2.resize(img,(int(resize_ratio*cols),300))
    #img_origin = img
    img_gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rects = []
    img_after_erode1 , rects1 = detect_letter_boxs_with_size(img, 1)
    img_after_erode2 , rects2 = detect_letter_boxs_with_size(img, 2.0/3.0)
    img_after_erode3 , rects3 = detect_letter_boxs_with_size(img, 1.0/3.0)
    rects += rects1
    rects = remove_duplicated_rects(rects, rects2)
    rects = remove_duplicated_rects(rects, rects3)
    img_contours_boxing = img_gray
    for rect in rects: # draw boxs to image
        box = cv2.boxPoints(rect.get_tuple())
        box = np.int0(box)
        img_contours_boxing = cv2.drawContours(img_contours_boxing,[box],0,(random.randrange(0,256),random.randrange(0,256),random.randrange(0,256)),1)
        #(random.randrange(0,256),random.randrange(0,256),random.randrange(0,256))
    """
    print "contours' len :",len(rects)
    for i, rect in enumerate(rects): # ocr text in each contours, and show that on origin_size
        rect.change_size(rows/300.0)
        rect.change_location(rows/300.0)
        crop_img = cv2.getRectSubPix(img, rect.get_tuple()[1], rect.get_tuple()[0])
        thresh, bw_img = cv2.threshold(crop_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_ocr = Image.fromarray(bw_img)
        txt = pytesseract.image_to_string(img_ocr, lang='Eng')
        print "text", i,":" ,txt
        while(2):
            cv2.imshow("cropped", bw_img)
            if cv2.waitKey(33) == 27:
                break
    """

    return [img_after_erode1, img_after_erode2, img_after_erode3, img_contours_boxing], rects, resize_ratio, padding

for i in range(20):
    img_origin = cv2.imread("C:/Users/min/PycharmProjects/MLPJ/posters/" + str(i) + ".jpg", 0)
    imgs, rects, resize_ratio, padding = detect_letter_boxs(img_origin)
    show_imgs(imgs)
    figures = get_figures_of_blocks(img_origin, rects)
    print_figures(figures)