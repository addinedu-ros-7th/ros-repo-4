import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 이미지 경로 설정
convert_img_path = "/home/zoo/addinedu/project/pyri/src/test/floor_img.png"

# 이미지 로드 및 그레이스케일 변환
map_img = cv2.imread(convert_img_path, cv2.IMREAD_GRAYSCALE)

def image_grid(img, grid_rows, grid_cols, threshold=210, 
               contact_distance_vertical=1, contact_distance_horizontal=1,
               continuous_distance_vertical=1, continuous_distance_horizontal=1):
    
    if len(img.shape) != 2:
        raise ValueError("입력 이미지는 흑백 이미지(numpy 2D 배열)여야 합니다.")
    if not (0 <= img.min() and img.max() <= 255):
        raise ValueError("입력 이미지의 값 범위는 0~255여야 합니다.")
    
    img_height, img_width = img.shape  
    cell_height = img_height // grid_rows  
    cell_width = img_width // grid_cols    

    map_cost = []  

    for row in range(grid_rows):
        row_values = []
        for col in range(grid_cols):
            cell_top = row * cell_height
            cell_left = col * cell_width
            cell_bottom = min(cell_top + cell_height, img_height)
            cell_right = min(cell_left + cell_width, img_width)

            cell = img[cell_top:cell_bottom, cell_left:cell_right]
            
            road = np.sum(cell > threshold)  
            wall = np.sum(cell <= threshold)  
            
            row_values.append(0 if road > wall else 1)
        
        map_cost.append(row_values)  

    for i in range(len(map_cost)):  
        start = -1  
        for j in range(len(map_cost[i])):
            if map_cost[i][j] == 1:  
                if start != -1:  
                    center_col = (start + j - 1) // 2  
                    if map_cost[i][center_col] != 1:  
                        map_cost[i][center_col] = 2  
                start = -1  
            elif start == -1:  
                start = j
        if start != -1:
            center_col = (start + len(map_cost[i]) - 1) // 2
            if map_cost[i][center_col] != 1:
                map_cost[i][center_col] = 2

    for j in range(len(map_cost[0])):  
        start = -1  
        for i in range(len(map_cost)):
            if map_cost[i][j] == 1:  
                if start != -1:  
                    center_row = (start + i - 1) // 2  
                    if map_cost[center_row][j] != 1:  
                        map_cost[center_row][j] = 2  
                start = -1  
            elif start == -1:  
                start = i
        if start != -1:
            center_row = (start + len(map_cost) - 1) // 2
            if map_cost[center_row][j] != 1:
                map_cost[center_row][j] = 2

    for i in range(len(map_cost)):
        for j in range(1, len(map_cost[i]) - 1):
            if map_cost[i][j] == 2:
                if all(map_cost[i + di][j] == 2 for di in range(-continuous_distance_vertical, continuous_distance_vertical + 1) if 0 <= i + di < len(map_cost)) or \
                   all(map_cost[i][j + dj] == 2 for dj in range(-continuous_distance_horizontal, continuous_distance_horizontal + 1) if 0 <= j + dj < len(map_cost[i])):
                    map_cost[i][j] = 0  

    for i in range(len(map_cost)):
        for j in range(len(map_cost[i])):
            if map_cost[i][j] == 2:
                for di in range(-contact_distance_vertical, contact_distance_vertical + 1):
                    for dj in range(-contact_distance_horizontal, contact_distance_horizontal + 1):
                        if 0 <= i + di < len(map_cost) and 0 <= j + dj < len(map_cost[i]):
                            if map_cost[i + di][j + dj] == 1:
                                map_cost[i][j] = 0
                                break
                    if map_cost[i][j] == 0:  
                        break
                    
    waypoints = [(i, j) for i in range(len(map_cost)) for j in range(len(map_cost[i])) if map_cost[i][j] == 2]

    num_clusters = len(waypoints) // 3  
    if num_clusters > 1:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(waypoints)
        new_map_cost = [[0 if cell == 2 else cell for cell in row] for row in map_cost]

        for cluster_center in kmeans.cluster_centers_:
            x, y = int(cluster_center[0]), int(cluster_center[1])
            new_map_cost[x][y] = 2  

        map_cost = new_map_cost

    return map_cost, cell_height, cell_width


###################################################################
# 입력 이미지 비율 및 픽셀값 조절
img_height, img_width = map_img.shape
resize_factor = 0.04  # 원하는 크기 비율 
resize_height = int(img_height * resize_factor)
resize_width = int(img_width * resize_factor)
map_img = cv2.resize(map_img, (resize_width, resize_height))

grid_rows = resize_height  # 세로 칸 수
grid_cols = resize_width  # 가로 칸 수
threshold = 100  # 픽셀 임계값
###################################################################

contact_distance_vertical = 2  
contact_distance_horizontal = 2  
continuous_distance_vertical = 1  
continuous_distance_horizontal = 1  

my_map, cell_height, cell_width = image_grid(map_img, grid_rows, grid_cols, threshold, 
                                             contact_distance_vertical, contact_distance_horizontal, 
                                             continuous_distance_vertical, continuous_distance_horizontal)

# OpenCV 창에 표시할 맵 생성
def draw_map(map_data, cell_height, cell_width, scale=1):
    map_image = np.zeros((len(map_data) * cell_height * scale, len(map_data[0]) * cell_width * scale), dtype=np.uint8)
    for i, row in enumerate(map_data):
        for j, cell in enumerate(row):
            top_left = (j * cell_width * scale, i * cell_height * scale)
            bottom_right = ((j + 1) * cell_width * scale, (i + 1) * cell_height * scale)
            if cell == 1:
                color = 255  # 벽
            elif cell == 2:
                color = 127  # 이동 경로
            else:
                color = 0  # 도로
            map_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = color
    return map_image

# 마우스 이벤트 핸들러
def mouse_callback(event, x, y, flags, param):
    global map_data, cell_height, cell_width, scale
    if event == cv2.EVENT_LBUTTONDOWN:
        col = x // (cell_width * scale)
        row = y // (cell_height * scale)
        if 0 <= row < len(map_data) and 0 <= col < len(map_data[0]):
            if map_data[row][col] == 2:
                map_data[row][col] = 0  # 이동경로(2) → 도로(0)
            elif map_data[row][col] == 0:
                map_data[row][col] = 2  # 도로(0) → 이동경로(2)

# 초기화
map_data = [row.copy() for row in my_map]  # 원본 데이터 유지
scale = 7

cv2.namedWindow("Grid Map Editor")
cv2.setMouseCallback("Grid Map Editor", mouse_callback)

while True:
    map_image = draw_map(map_data, cell_height, cell_width, scale)
    cv2.imshow("Grid Map Editor", map_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 13:  # 'q' 또는 ESC 키
        break

cv2.destroyAllWindows()

# 결과 맵 변환 및 저장
scaled_map = np.array(map_data, dtype=np.uint8)
scaled_map = np.where(scaled_map == 1, 255, scaled_map)  
scaled_map = np.where(scaled_map == 2, 127, scaled_map)  

loaded_image = np.flipud(scaled_map)

# 수정된 맵을 시각화
plt.figure(figsize=(7, 7))
plt.imshow(loaded_image, cmap="gray", vmin=0, vmax=255, origin='lower')
plt.title("Edited Waypoints Map")

# 축 제거
plt.xticks([])  
plt.yticks([])  

plt.show()

map_data = np.zeros_like(loaded_image)
map_data[loaded_image == 255] = 1  
map_data[loaded_image == 127] = 2  

path_coordinates = [(row_idx, col_idx) for row_idx, row in enumerate(map_data) for col_idx, cell in enumerate(row) if cell == 2]

origin_min = (6, 6)  
origin_max = (75, 31)
target_min = (0, 0)
target_max = (1.5, 3.4)

scale_x = (target_max[0] - target_min[0]) / (origin_max[1] - origin_min[1])  
scale_y = (target_max[1] - target_min[1]) / (origin_max[0] - origin_min[0])  

def transform_coordinates(pixel):
    row, col = pixel
    new_x = (col - origin_min[1]) * scale_x  
    new_y = (row - origin_min[0]) * scale_y  
    return (round(new_x, 2), round(new_y, 2))

transformed_path_coordinates = [transform_coordinates(coord) for coord in path_coordinates]

print("변환된 이동 경로 좌표 리스트:", transformed_path_coordinates)
print("좌표 개수:", len(transformed_path_coordinates))

plt.figure(figsize=(8, 8))
plt.imshow(loaded_image, cmap="gray", vmin=0, vmax=255, origin='lower')

for (orig, (new_x, new_y)) in zip(path_coordinates, transformed_path_coordinates):
    plt.text(orig[1], orig[0], f"({new_x:.2f},{new_y:.2f})", color="red", fontsize=6, ha="center", va="center", 
             bbox=dict(facecolor='white', alpha=1, pad = 2))
    
plt.title("Waypoints Map")

# 축 제거
plt.xticks([])  
plt.yticks([])  

# 저장
waypoints_map_path = "/home/zoo/addinedu/project/pyri/src/test/test_map.png"
plt.savefig(waypoints_map_path, dpi=300)  
plt.show()
