import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import math

# 공통함수
def img_show(img):
    plt.figure(figsize=(8,6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
def count_in_color(img_array):
    # 처음에 BGR로 집어넣기
    r, g, b = [], [], []
    for x in img_array:
        for y in x:
            b.append(y[0])
            g.append(y[1])
            r.append(y[2])
            
    rgb = list(set(list(zip(r,g,b))))
    return (len(rgb))
    
# ground truth 3개 rgb 색상으로 바꾸기 및 색상별 좌표 저장 [(),()] 형식
def switch_3(img_array):
    img_gt_ = img_array.copy()
    white = np.array((255,255,255))
    black = np.array((0,0,0))
    blue = np.array((0,0,255))
    color_l = [[255,255,255],[0,0,0],[0,0,255]]
    gt_white, gt_black, gt_blue = [], [], []

    for i in range(len(img_gt_)):
        for j in range(len(img_gt_[i])):
            temp = (np.linalg.norm(img_gt_[i][j]-white),np.linalg.norm(img_gt_[i][j]-black),np.linalg.norm(img_gt_[i][j]-blue))
            temp_idx = temp.index(min(temp))

            if list(img_gt_[i][j]) != color_l[temp_idx]:
                img_gt_[i][j]=color_l[temp_idx]
                
            if list(img_gt_[i][j]) == [0,0,255]:
                gt_blue.append((j,i))
            elif list(img_gt_[i][j]) == [0,0,0]:
                gt_black.append((j,i))
            else:
                gt_white.append((j,i))
                
    return(img_gt_, gt_black, gt_blue, gt_white)



def max_cnt(img):
    # rgb image 들어오면 cnt가 나감
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).copy()
    
    # Threshold K 설정 및 적용 (255 : 흰색 / 0 : 검정색)
    k=200
    img_gray = np.where(img_gray<k, 0, img_gray)
    img_gray = np.where(img_gray>=k, 255, img_gray)
    
    # Contour 적용
    img_gray = cv2.bitwise_not(img_gray)
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    
    # max Contour (img_c_max)
    area_res=[]
    for i in range(len(contours)):
        cnt = contours[i]
        area_res.append(cv2.contourArea(cnt))

    max_idx=area_res.index(max(area_res))
    cnt = contours[max_idx]
    
    return(cnt)

def ret(point,pad):
    if point+pad<=0:
        return point
    else:
        return point+pad
    
def cor_iou(gt, pred):
    gt_=set(gt)
    pred_=set(pred)
    
    union=gt_|pred_
    inter=gt_&pred_
    return (len(inter) / len(union))

# distance, triangle area function
def distance(x1, y1, x2, y2):
    result = math.sqrt( math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return result

def triangle(a,b,c):
    s = (a+b+c)/2
    res = math.sqrt(s*(s-a)*(s-b)*(s-c))
    return(res)

def location_cal(p, q, t):
    if p[0] - q[0] == 0:
        return (t[0] - p[0])
    else:
        return (t[1] - ((p[1] - q[1]) / (p[0] - q[0])) * (t[0] - p[0]) - p[1])

def inout(a, b, c, d, point):
    if (b[0] - d[0]) != 0 and (a[0] - c[0]) != 0:
        if location_cal(a, b, point) >= 0 and location_cal(c, d, point) <= 0 and ((b[1] - d[1]) / (b[0] - d[0])) * (
        location_cal(b, d, point)) >= 0 and ((a[1] - c[1]) / (a[0] - c[0])) * (location_cal(a, c, point)) <= 0:
            return (True)
        else:
            return (False)
    elif (b[0] - d[0]) == 0 and (a[0] - c[0]) != 0:
        if location_cal(a, b, point) >= 0 and location_cal(c, d, point) <= 0 and (location_cal(b, d, point)) <= 0 and (
                (a[1] - c[1]) / (a[0] - c[0])) * (location_cal(a, c, point)) <= 0:
            return (True)
        else:
            return (False)
    elif (b[0] - d[0]) != 0 and (a[0] - c[0]) == 0:
        if location_cal(a, b, point) >= 0 and location_cal(c, d, point) <= 0 and ((b[1] - d[1]) / (b[0] - d[0])) * (
        location_cal(b, d, point)) >= 0 and (location_cal(a, c, point)) >= 0:
            return (True)
        else:
            return (False)
    elif (b[0] - d[0]) == 0 and (a[0] - c[0]) == 0:
        if location_cal(a, b, point) >= 0 and location_cal(c, d, point) <= 0 and (location_cal(b, d, point)) <= 0 and (
        location_cal(a, c, point)) >= 0:
            return (True)
        else:
            return (False)

def hed_near(img_arr, k, show_tf=False, is_blue=False):
    # 0_plot은 꼭지점
    a_plot = (0, 0)
    d_plot = (img_arr.shape[1], img_arr.shape[0])
    b_plot = (d_plot[0], 0)
    c_plot = (0, d_plot[1])

    # 각 꼭지점에서 가장 가까운 점을 찾기 위해 df에 관련 내용 대입
    df = pd.DataFrame(columns=['x', 'y', 'a', 'b', 'c', 'd'])

    for x_ in range(img_arr.shape[1]):
        for y_ in range(img_arr.shape[0]):
            if is_blue:
                if 100<list(img_arr[y_][x_])[0]+list(img_arr[y_][x_])[1]+list(img_arr[y_][x_])[2]<300:
            # if sum(list(img_arr[y_][x_]))/3 >=k:
                #plt.scatter(x_, y_, s=75)

                    df = df.append({'x': x_, 'y': y_,
                                    'a': distance(x_, y_, a_plot[0], a_plot[1])
                                       , 'b': distance(x_, y_, b_plot[0], b_plot[1])
                                       , 'c': distance(x_, y_, c_plot[0], c_plot[1])
                                       , 'd': distance(x_, y_, d_plot[0], d_plot[1])
                                    }, ignore_index=True)
            else:
                if list(img_arr[y_][x_])[0]>=k & list(img_arr[y_][x_])[1]>=k & list(img_arr[y_][x_])[2]>=k:
            # if sum(list(img_arr[y_][x_]))/3 >=k:
                #plt.scatter(x_, y_, s=75)

                    df = df.append({'x': x_, 'y': y_,
                                    'a': distance(x_, y_, a_plot[0], a_plot[1])
                                       , 'b': distance(x_, y_, b_plot[0], b_plot[1])
                                       , 'c': distance(x_, y_, c_plot[0], c_plot[1])
                                       , 'd': distance(x_, y_, d_plot[0], d_plot[1])
                                    }, ignore_index=True)

    # 0_pred는 각 꼭지점에서 가장 가까운 edge의 끝점들 표시한 점
    a_pred = (df.sort_values(by=['a']).iloc[0][0], df.sort_values(by=['a']).iloc[0][1])
    b_pred = (df.sort_values(by=['b']).iloc[0][0], df.sort_values(by=['b']).iloc[0][1])
    c_pred = (df.sort_values(by=['c']).iloc[0][0], df.sort_values(by=['c']).iloc[0][1])
    d_pred = (df.sort_values(by=['d']).iloc[0][0], df.sort_values(by=['d']).iloc[0][1])

    # 전체 이미지 중 해당하는 부분의 ratio를 구하는 과정
    l1 = distance(a_pred[0], a_pred[1], b_pred[0], b_pred[1])
    l2 = distance(b_pred[0], b_pred[1], d_pred[0], d_pred[1])
    l3 = distance(d_pred[0], d_pred[1], c_pred[0], c_pred[1])
    l4 = distance(c_pred[0], c_pred[1], a_pred[0], a_pred[1])
    d1 = distance(a_pred[0], a_pred[1], d_pred[0], d_pred[1])
    d2 = distance(b_pred[0], b_pred[1], c_pred[0], c_pred[1])

    area_full = img_arr.shape[0] * img_arr.shape[1]
    area_pred = int(triangle(l1, l2, d1) + triangle(l3, l4, d1))
    ratio_pred = area_pred / area_full
    
    a, b, c, d = a_pred, b_pred, c_pred, d_pred
    
    hed_in_cor = []
    hed_in_cor_x = []
    hed_in_cor_y = []
    for x in range(img_arr.shape[1]):
        for y in range(img_arr.shape[0]):
            if inout(a, b, c, d, (x, y)):
                hed_in_cor.append((x, y))
                hed_in_cor_x.append(x)
                hed_in_cor_y.append(y)
    
    return (hed_in_cor)    


##########################################
# 1. Max contour 방식
    
def max_contour(path, show=False):
    #path = './data_edge_eval/2.jpg'
    img_path = path.split('.jpg')[0]

    # 이미지 로드
    img_edge = cv2.imread(img_path + '_dexi.png')
    contour_res = np.ones((img_edge.shape), dtype=np.uint8)*255

    # pickle 그대로 불러오기 >> img_pred / 필요한 정보 df 형태로 >> img_pred_df
    img_pred = pd.read_pickle(img_path + '_pred_detail.pickle')

    label_idx = img_pred[3][0]
    img_pred_df = pd.DataFrame(img_pred[0][0][0:label_idx], columns=['y1','x1','y2','x2']).copy()
    img_pred_df['class'] = img_pred[2][0][0:label_idx]
    img_pred_df = img_pred_df.astype('int')
    img_pred_df['score'] = img_pred[1][0][0:label_idx]
    img_pred_df.sort_values(by=['class'], inplace=True)

    # 건물 전체 contour
    full_cnt = max_cnt(img_edge)
    cv2.drawContours(contour_res, [full_cnt], 0, (0, 0, 0), cv2.FILLED)

    # 창문 contour 하나씩 적용
    pad=5
    for i in range(len(img_pred_df)):
        if img_pred_df['class'].iloc[i] == 0:
            y1,x1,y2,x2,_,_ = img_pred_df.iloc[i]
            temp_crop = img_edge[ret(int(y1),-pad):ret(int(y2),+pad), ret(int(x1),-pad):ret(int(x2),+pad)].copy()
            crop_cnt = max_cnt(temp_crop)
            crop_cnt_move = np.array([[[x[0][0]+int(x1)-pad, x[0][1]+int(y1)-pad]] for x in crop_cnt.tolist()]).copy()
            cv2.drawContours(contour_res, [crop_cnt_move], 0, (0,0,255), cv2.FILLED)

        if img_pred_df['class'].iloc[i] != 0 and img_pred_df['class'].iloc[i] != 10:
            y1,x1,y2,x2,_,_ = img_pred_df.iloc[i]
            temp_crop = img_edge[ret(int(y1),-pad):ret(int(y2),+pad), ret(int(x1),-pad):ret(int(x2),+pad)].copy()
            crop_cnt = max_cnt(temp_crop)
            crop_cnt_move = np.array([[[x[0][0]+int(x1)-pad, x[0][1]+int(y1)-pad]] for x in crop_cnt.tolist()]).copy()
            cv2.drawContours(contour_res, [crop_cnt_move], 0, (255,255,255), cv2.FILLED)

    pred_white, pred_black, pred_blue = [], [], []
    test_p = contour_res.tolist().copy()
            
    for y in range(len(test_p)):
        for x in range(len(test_p[y])):
            if list(test_p[y][x]) == [0,0,255]:
                pred_blue.append([x,y])
            elif list(test_p[y][x]) == [0,0,0]:
                pred_black.append([x,y])
            else:
                pred_white.append([x,y])
            
    if show:
        img_show(contour_res)
        print('창문 : {}'.format(len(pred_blue)))
        print('외벽 : {}'.format(len(pred_black)))
        print('배경 : {}'.format(len(pred_white)))
        
    return (contour_res, pred_blue, pred_black, pred_white)


###############################################
# 2. Near point 방식

def near_point(path, show=False):
    #path = './data_edge_eval/2.jpg'
    img_path = path.split('.jpg')[0]

    # 이미지 로드
    img_edge = cv2.imread(img_path + '_hed.png')
    hed_map_res = np.ones((img_edge.shape), dtype=np.uint8)*255

    # pickle 그대로 불러오기 >> img_pred / 필요한 정보 df 형태로 >> img_pred_df
    img_pred = pd.read_pickle(img_path + '_pred_detail.pickle')

    label_idx = img_pred[3][0]
    img_pred_df = pd.DataFrame(img_pred[0][0][0:label_idx], columns=['y1','x1','y2','x2']).copy()
    img_pred_df['class'] = img_pred[2][0][0:label_idx]
    img_pred_df = img_pred_df.astype('int')
    img_pred_df['score'] = img_pred[1][0][0:label_idx]
    img_pred_df.sort_values(by=['class'], inplace=True)

    res = hed_near(img_edge, 200)
    for x_,y_ in res:
        hed_map_res[y_][x_] = [0,0,0]

    pad=5

    for i in tqdm(range(len(img_pred_df))):
        if img_pred_df['class'].iloc[i] == 0:
            y1,x1,y2,x2,_,_ = img_pred_df.iloc[i]
            temp_crop = img_edge[ret(int(y1),-pad):ret(int(y2),+pad), ret(int(x1),-pad):ret(int(x2),+pad)].copy()
            temp_crop_res = hed_near(temp_crop, 100)
            temp_crop_res_move = [(x[0]+int(x1)-pad, x[1]+int(y1)-pad) for x in temp_crop_res].copy()

            for x_,y_ in temp_crop_res_move:
                hed_map_res[y_][x_] = [0,0,255]

        if img_pred_df['class'].iloc[i] != 0 and img_pred_df['class'].iloc[i] != 10:
            y1,x1,y2,x2,_,_ = img_pred_df.iloc[i]
            temp_crop = img_edge[ret(int(y1),-pad):ret(int(y2),+pad), ret(int(x1),-pad):ret(int(x2),+pad)].copy()
            temp_crop_res = hed_near(temp_crop, 100)
            temp_crop_res_move = [(x[0]+int(x1)-pad, x[1]+int(y1)-pad) for x in temp_crop_res].copy()

            for x_,y_ in temp_crop_res_move:
                hed_map_res[y_][x_] = [255,255,255]

    pred_white, pred_black, pred_blue = [], [], []
    test_p = hed_map_res.tolist().copy()
            
    for y in range(len(test_p)):
        for x in range(len(test_p[y])):
            if list(test_p[y][x]) == [0,0,255]:
                pred_blue.append([x,y])
            elif list(test_p[y][x]) == [0,0,0]:
                pred_black.append([x,y])
            else:
                pred_white.append([x,y])
            
    if show:
        img_show(hed_map_res)
        print('창문 : {}'.format(len(pred_blue)))
        print('외벽 : {}'.format(len(pred_black)))
        print('배경 : {}'.format(len(pred_white)))
        
    return (hed_map_res, pred_blue, pred_black, pred_white)

#######################################################
# 3. Mixed 방식

def fused(path, show=False):
    #path = './data_edge_eval/2.jpg'
    img_path = path.split('.jpg')[0]

    # 이미지 로드
    img_edge = cv2.imread(img_path + '_dexi.png')
    contour_res = np.ones((img_edge.shape), dtype=np.uint8)*255

    # pickle 그대로 불러오기 >> img_pred / 필요한 정보 df 형태로 >> img_pred_df
    img_pred = pd.read_pickle(img_path + '_pred_detail.pickle')

    label_idx = img_pred[3][0]
    img_pred_df = pd.DataFrame(img_pred[0][0][0:label_idx], columns=['y1','x1','y2','x2']).copy()
    img_pred_df['class'] = img_pred[2][0][0:label_idx]
    img_pred_df = img_pred_df.astype('int')
    img_pred_df['score'] = img_pred[1][0][0:label_idx]
    img_pred_df.sort_values(by=['class'], inplace=True)

    # 건물 전체 contour
    full_cnt = max_cnt(img_edge)
    cv2.drawContours(contour_res, [full_cnt], 0, (0, 0, 0), cv2.FILLED)

    pad=5

    for i in tqdm(range(len(img_pred_df))):
        if img_pred_df['class'].iloc[i] == 0:
            y1,x1,y2,x2,_,_ = img_pred_df.iloc[i]

            contour_res_copy = contour_res.copy()
            temp_crop = img_edge[ret(int(y1),-pad):ret(int(y2),+pad), ret(int(x1),-pad):ret(int(x2),+pad)].copy()
            crop_cnt = max_cnt(temp_crop)
            crop_cnt_move = np.array([[[x[0][0]+int(x1)-pad, x[0][1]+int(y1)-pad]] for x in crop_cnt.tolist()]).copy()
            cv2.drawContours(contour_res_copy, [crop_cnt_move], 0, (0,0,255), cv2.FILLED)

            temp_crop = contour_res_copy[ret(int(y1),-pad):ret(int(y2),+pad), ret(int(x1),-pad):ret(int(x2),+pad)].copy()
            temp_crop_res = hed_near(temp_crop, 100, is_blue=True)
            temp_crop_res_move = [(x[0]+int(x1)-pad, x[1]+int(y1)-pad) for x in temp_crop_res].copy()

            for x_,y_ in temp_crop_res_move:
                contour_res[y_][x_] = [0,0,255]

        if img_pred_df['class'].iloc[i] != 0 and img_pred_df['class'].iloc[i] != 10:
            y1,x1,y2,x2,_,_ = img_pred_df.iloc[i]
            temp_crop = img_edge[ret(int(y1),-pad):ret(int(y2),+pad), ret(int(x1),-pad):ret(int(x2),+pad)].copy()
            crop_cnt = max_cnt(temp_crop)
            crop_cnt_move = np.array([[[x[0][0]+int(x1)-pad, x[0][1]+int(y1)-pad]] for x in crop_cnt.tolist()]).copy()
            cv2.drawContours(contour_res, [crop_cnt_move], 0, (255,255,255), cv2.FILLED)

    pred_white, pred_black, pred_blue = [], [], []
    test_p = contour_res.tolist().copy()
            
    for y in range(len(test_p)):
        for x in range(len(test_p[y])):
            if list(test_p[y][x]) == [0,0,255]:
                pred_blue.append([x,y])
            elif list(test_p[y][x]) == [0,0,0]:
                pred_black.append([x,y])
            else:
                pred_white.append([x,y])
            
    if show:
        img_show(contour_res)
        print('창문 : {}'.format(len(pred_blue)))
        print('외벽 : {}'.format(len(pred_black)))
        print('배경 : {}'.format(len(pred_white)))
        
    return (contour_res, pred_blue, pred_black, pred_white)









#######################################################################################
# 열 누수 탐지

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from collections import Counter
import pickle

def u_dist(a,b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2

def near_class_idx3(point,lst):
    score=list()
    for l in lst:
        score.append(u_dist(point, l))
    near_idx=score.index(min(score))
    return near_idx

def count_pixel_class(img, lst, show=False):
    #이미지 픽셀 RGB값 리스트화
    img_px_lst=[]
    if show == True:
        print("Image Pixel Save...")
        for x in tqdm(range(img.size[0])):
            for y in range(img.size[1]):
                r,g,b=img.getpixel((x,y))
                img_px_lst.append((r,g,b))

        img_px_class=[]
        print("Find Pixel Class...")

        for item in tqdm(img_px_lst):
            img_px_class.append(near_class_idx3(item ,lst))
            
    else:
        #print("Image Pixel Save...")
        for x in range(img.size[0]):
            for y in range(img.size[1]):
                r,g,b=img.getpixel((x,y))
                img_px_lst.append((r,g,b))

        img_px_class=[]
        #print("Find Pixel Class...")

        for item in img_px_lst:
            img_px_class.append(near_class_idx3(item ,lst))

    return img_px_class

def find_threshold(lst):
    lst_count=Counter([x for x in lst if x >= 150])
    lst_count_sort= sorted(lst_count)

    temp_class_list = [0 for _ in range(279)]
    for x in lst:
        temp_class_list[x] += 1

    first_max, second_max = lst_count.most_common(2)[0][0], lst_count.most_common(2)[1][0]
                
    count_min = min(temp_class_list[min(first_max,second_max) : max(first_max,second_max)])
    results = []
    for i in range(len(temp_class_list)):
        if min(first_max,second_max) <= i < max(first_max,second_max):
            if temp_class_list[i]==count_min:
                results.append(i)
                
    result = int(sum(results)/len(results))
    return result


#########################################################
#########################################################
# 1. Bimodal Analysis


def bimdoal_analysis(path):
    # 0. 상대적 온도 비교를 위한 온도 class 설정 (온도 스펙트럼 바 및 색상 기준)

    bar_image_path = './thermal_bar.jpg'
    bar_im = Image.open(bar_image_path)
    rgb_bar_img = bar_im.convert('RGB')

    bar_r_lst, bar_g_lst, bar_b_lst=[], [], []

    for y in range(bar_im.size[1]):
        r,g,b = rgb_bar_img.getpixel((1,bar_im.size[1]-y-1))
        bar_r_lst.append(r)
        bar_g_lst.append(g)
        bar_b_lst.append(b)

    bar_rgb_lst = list(zip(bar_r_lst,bar_g_lst,bar_b_lst))
    
    #path = './data_heatloss/0001.jpg'
    img_path = path.split('.jpg')[0]

    # 이미지 로드
    img_infrared = cv2.imread(img_path + '.jpg')
    img_infrared = cv2.cvtColor(img_infrared, cv2.COLOR_BGR2RGB).copy()
    img_edge = cv2.imread(img_path + 'g_dexi.png')


    # pickle 그대로 불러오기 >> img_pred / 필요한 정보 df 형태로 >> img_pred_df
    img_pred = pd.read_pickle(img_path + 'g_pred_detail.pickle')

    label_idx = img_pred[3][0]
    img_pred_df = pd.DataFrame(img_pred[0][0][0:label_idx], columns=['y1','x1','y2','x2']).copy()
    img_pred_df['class'] = img_pred[2][0][0:label_idx]
    img_pred_df = img_pred_df.astype('int')
    img_pred_df['score'] = img_pred[1][0][0:label_idx]
    img_pred_df.sort_values(by=['class'], inplace=True)
    img_pred_df = img_pred_df[img_pred_df['class']==0]

    full_window_class = []
    pad = 5
    for i in range(len(img_pred_df)):
        contour_res = np.ones((img_edge.shape), dtype=np.uint8)*255
        contour_res_black = np.zeros((img_edge.shape), dtype=np.uint8)
        y1,x1,y2,x2,_,_ = img_pred_df.iloc[i]

        contour_res_copy = contour_res.copy()
        temp_crop = img_edge[ret(int(y1),-pad):ret(int(y2),+pad), ret(int(x1),-pad):ret(int(x2),+pad)].copy()
        crop_cnt = max_cnt(temp_crop)
        crop_cnt_move = np.array([[[x[0][0]+int(x1)-pad, x[0][1]+int(y1)-pad]] for x in crop_cnt.tolist()]).copy()
        cv2.drawContours(contour_res_copy, [crop_cnt_move], 0, (0,0,255), cv2.FILLED)

        temp_crop = contour_res_copy[ret(int(y1),-pad):ret(int(y2),+pad), ret(int(x1),-pad):ret(int(x2),+pad)].copy()
        temp_crop_res = hed_near(temp_crop, 100, is_blue=True)
        temp_crop_res_move = [(x[0]+int(x1)-pad, x[1]+int(y1)-pad) for x in temp_crop_res].copy()

        img_px_class=[]
        for x_,y_ in temp_crop_res_move:
            # 시각화
            contour_res_black[y_][x_] = [255,255,255]
            # 온도분석
            r,g,b = img_infrared[y_][x_]
            img_px_class.append(near_class_idx3((r,g,b), bar_rgb_lst))

        full_window_class.append(img_px_class)


        plt.figure(figsize=(5,5))
        plt.title('{} window area'.format(i+1))
        plt.imshow(img_infrared)
        plt.imshow(contour_res_black, alpha=0.3)
        plt.axis('off')
        plt.show()

    full_window_class_flat = sum(full_window_class, [])
    thres_class = find_threshold(full_window_class_flat)
    print('threshold : {}'.format(thres_class))


    plt.bar(Counter(full_window_class_flat).keys(), Counter(full_window_class_flat).values())
    plt.title('all window temp dist')
    plt.axvline(x=thres_class, color='r', linestyle='--', linewidth=2)
    plt.xlim(0,279)
    plt.xlabel('temp class')
    plt.ylabel('pixel count')
    plt.show()

    for i in range(len(full_window_class)):
        plt.bar(Counter(full_window_class[i]).keys(), Counter(full_window_class[i]).values())
        plt.title('{} window temp dist (thres={})'.format(i+1, thres_class))
        plt.axvline(x=thres_class, color='r', linestyle='--', linewidth=2)
        plt.xlim(0,279)
        plt.ylim(0,max(Counter(full_window_class_flat).values()))
        plt.xlabel('temp class')
        plt.ylabel('pixel count')
        plt.show()
        cnt = len([x for x in full_window_class[i] if x >= thres_class])

        print('[Threshold 기반 분석]')
        print('이상 픽셀 수 : {}'.format(cnt))
        print('이상 픽셀 비율 : {:.3f}'.format(cnt / len(full_window_class[i]) * 100))
        
        
        
        
##################################################################
# Delta T analysis


def deltaT_analysis(path):
    #path = './data_heatloss/0001.jpg'
    
    bar_image_path = './thermal_bar.jpg'
    bar_im = Image.open(bar_image_path)
    rgb_bar_img = bar_im.convert('RGB')

    bar_r_lst, bar_g_lst, bar_b_lst=[], [], []

    for y in range(bar_im.size[1]):
        r,g,b = rgb_bar_img.getpixel((1,bar_im.size[1]-y-1))
        bar_r_lst.append(r)
        bar_g_lst.append(g)
        bar_b_lst.append(b)

    bar_rgb_lst = list(zip(bar_r_lst,bar_g_lst,bar_b_lst))
    
    img_path = path.split('.jpg')[0]

    full_img = Image.open(path)
    rgb_full_img = full_img.convert('RGB')

    with open(path.split('.jpg')[0] + 'g_pred_detail.pickle', "rb") as fh:
        data = pickle.load(fh)
    res = data

    upper_white = np.array([255,255,255])
    lower_black = np.array([0,0,0])

    Delta_T = []

    for i in tqdm(range(res[3][0])):
        if res[2][0][i] == 0:
            y1,x1,y2,x2 = res[0][0][i]

            cropped = full_img.crop((x1,y1,x2,y2))

            margin_y, margin_x = ((res[0][0][i][2] - res[0][0][i][0]))/10, ((res[0][0][i][3]-res[0][0][i][1]))/10

            margin_cropped = full_img.crop((x1-margin_x, y1-margin_y, x2+margin_x, y2+margin_y))

            mask_cropped = cropped.crop((cropped.size[0]/2 - margin_cropped.size[0]/2, cropped.size[1]/2 - margin_cropped.size[1]/2, cropped.size[0]/2 - margin_cropped.size[0]/2  + margin_cropped.size[0], cropped.size[1]/2 - margin_cropped.size[1]/2 + margin_cropped.size[1]))

            mask_cropped_array = np.array(mask_cropped)
            margin_cropped_array = np.array(margin_cropped)

            masking_array = mask_cropped_array

            for i in range(len(mask_cropped_array)):
                for j in range(len(mask_cropped_array[0])):
                    if (masking_array[i][j][0]==0) & (masking_array[i][j][1]==0) & (masking_array[i][j][2]==0):
                        masking_array[i][j] = upper_white


            for i in range(len(mask_cropped_array)):
                for j in range(len(mask_cropped_array[0])):
                    if (masking_array[i][j][0]<255) or (masking_array[i][j][1]<255) or (masking_array[i][j][2]<255):
                        masking_array[i][j] = lower_black


            background = cv2.copyTo(margin_cropped_array, masking_array, dst=None)
            mask_cropped_array = np.array(mask_cropped)


            mask_res = count_pixel_class(mask_cropped,  bar_rgb_lst)
            ground_image=Image.fromarray(background)

            background_res = count_pixel_class(ground_image, bar_rgb_lst)

            len_mask = 0
            for i in mask_res:
                if i != 0:
                    len_mask +=1

            len_background = 0
            for i in background_res:
                if i != 0:
                    len_background +=1

            sum_window_T = 0
            for i in range(len(mask_res)):
                sum_window_T += mask_res[i]

            window_avg = sum_window_T/len_mask



            sum_envelope_T = 0
            for i in range(len(background_res)):
                sum_envelope_T += background_res[i]

            envelope_avg = sum_envelope_T/len_background

            delta_T = window_avg - envelope_avg

            Delta_T.append(delta_T)
            
            
    print("Delta T analysis")
    for i in range(len(Delta_T)):
        print(" - {} window : {:.3f}".format(i, Delta_T[i]))
    return(Delta_T)