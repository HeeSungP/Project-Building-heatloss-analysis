import numpy as np
import cv2

def perspective_transform(path):
    point_list = []
    count = 0

    # 원본 이미지
    #path = './data_raw/cap5.jpg'
    img_original = cv2.imread(path)
    img_original_2 = img_original.copy()

    def mouse_callback(event, x, y, flags, param):
        #global point_list, count, img_original, ct
        #ct=0

        # 마우스 왼쪽 버튼 누를 때마다 좌표를 리스트에 저장
        if event == cv2.EVENT_LBUTTONDOWN:
            point_list.append((x, y))
            cv2.circle(img_original, (x, y), 3, (0, 0, 255), -1)
            #print(point_list)
            
            if len(point_list)==2:
                cv2.line(img_original, point_list[0], point_list[1], (0,255,0), 2)
            if len(point_list)==3:
                cv2.line(img_original, point_list[0], point_list[2], (0,255,0), 2)
            if len(point_list)==4:
                cv2.line(img_original, point_list[3], point_list[1], (0,255,0), 2)
                cv2.line(img_original, point_list[3], point_list[2], (0,255,0), 2)
            

    def mouse_callback_2(event, x, y, flags, param):
        #global point_list, count, img_original_3

        # 마우스 왼쪽 버튼 누를 때마다 좌표를 리스트에 저장
        if event == cv2.EVENT_LBUTTONDOWN:
            point_list.append((x, y))
            cv2.circle(img_original_3, (x, y), 3, (255, 0, 0), -1)

    # 원근변환
    cv2.namedWindow('perspective transform')
    cv2.setMouseCallback('perspective transform', mouse_callback)

    while(True):
        cv2.imshow("perspective transform", img_original)
        height, width = img_original.shape[:2]


        if cv2.waitKey(1)&0xFF == 32: # spacebar를 누르면 루프에서 빠져나옴 + 창종료
            cv2.destroyAllWindows()
            break

    try:
        # 좌표 순서 - 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝
        pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
        pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    except:
        print("원근변환을 위한 좌표를 잘못 선택하셨습니다. 다시 실행해 주세요. (Red)")

    M = cv2.getPerspectiveTransform(pts1,pts2)
    img_result = cv2.warpPerspective(img_original_2, M, (width,height))


    # 거리재는 위치 선택
    img_original_3 = img_result.copy()

    cv2.namedWindow('end point selection')
    cv2.setMouseCallback('end point selection', mouse_callback_2)
    #cv2.imshow("end point selection", img_result)

    while(True):
        cv2.imshow("end point selection", img_original_3)
        height, width = img_original_3.shape[:2]
        if cv2.waitKey(1)&0xFF == 32: # spacebar를 누르면 루프에서 빠져나옴 + 창종료
            cv2.destroyAllWindows()
            break

    try:
        #print("##########################")
        #print("선택 좌표 : {}, {}".format(point_list[4],point_list[5]))
        print("직선에 존재하는 픽셀 수 : {}".format(abs(point_list[4][0]-point_list[5][0])))
    except:
        print('직선 픽셀 수 계산을 위한 좌표를 잘못 선택하였습니다. 다시 실행해 주세요. (Blue)')

    cv2.imwrite("./data/"+path.split('/')[-1],img_result)
    cv2.imwrite("./data_res/"+path.split('/')[-1],img_result)
    print('변환 이미지 저장 : {}'.format("./data/"+path.split('/')[-1]))
    
    return(abs(point_list[4][0]-point_list[5][0]))