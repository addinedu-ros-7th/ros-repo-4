import cv2
import numpy as np
from scipy.spatial import distance
import math
import string
import itertools
import heapq

def find_parent(parent, i):
    if parent[i] == i:
        return i
    return find_parent(parent, parent[i])

def union(parent, rank, i, j):
    i_id = find_parent(parent, i)
    j_id = find_parent(parent, j)
    if i_id != j_id:
        if rank[i_id] > rank[j_id]:
            parent[j_id] = i_id
        elif rank[i_id] < rank[j_id]:
            parent[i_id] = j_id
        else:
            parent[j_id] = i_id
            rank[i_id] += 1
        return True
    return False

def kruskal(graph):
    result = []
    i = 0
    e = 0
    graph['edges'] = sorted(graph['edges'], key=lambda item: item[2])
    parent = []
    rank = []

    for _ in range(graph['num_nodes']):
        parent.append(_)
        rank.append(0)

    while e < graph['num_nodes'] - 1:
        u, v, w = graph['edges'][i]
        i += 1
        u_index = graph['nodes'].index(u)
        v_index = graph['nodes'].index(v)

        if union(parent, rank, u_index, v_index):
            e += 1
            result.append((u, v, w))

    return result

def generate_string_list():
  """A부터 ZZ까지 문자열을 생성하는 함수"""
  result = []
  for length in range(1, 3):  # 1은 A, 2는 AA, AB, ... , AZ, BA, BB, ..., BZ, ..., ZZ
    for s in itertools.product(string.ascii_uppercase, repeat=length):
      result.append("".join(s))
  return result

# print(string_list)
def astar(graph, start, end):
    """A* 알고리즘을 사용하여 최단 경로를 찾는 함수"""
    open_set = [(0, start)]  # (f_score, node) 튜플을 open_set에 저장
    came_from = {}  # 각 노드의 이전 노드를 저장하는 딕셔너리
    g_score = {node: float('inf') for node in graph}  # 각 노드의 g_score (시작 노드로부터의 실제 거리)
    g_score[start] = 0

    while open_set:
        current_f_score, current_node = heapq.heappop(open_set)

        if current_node == end:  # 목표 노드에 도착
            path = []
            while current_node in came_from:
                path.insert(0, current_node)
                current_node = came_from[current_node]
            path.insert(0, start)
            return path, g_score[end]

        for neighbor, weight in graph[current_node].items(): # 수정: graph[current_node]의 items() 메서드 사용
            tentative_g_score = g_score[current_node] + weight  # 현재 노드를 거쳐 이웃 노드까지의 g_score

            if tentative_g_score < g_score[neighbor]:  # 더 짧은 경로를 찾았다면 갱신
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score  # f_score 계산 (g_score + h_score)
                heapq.heappush(open_set, (f_score, neighbor))  # open_set에 추가

    return None, None  # 경로를 찾지 못함


def find_connect_nodes(mst, node):
    cnt = 0
    connected_nodes = []
    for val in mst:
        # print( val[0], node)
        if val[0] == node:
            cnt += 1
            connected_nodes.append(val[1])

        if val[1] == node:
            cnt += 1
            connected_nodes.append(val[0])
    return cnt, connected_nodes


def load_map(image_path):
    loaded_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # start_point = (26, 67)  # 시작점 좌표 (x, y)
    # end_point = (36, 67)  # 끝점 좌표 (x, y)
    # color = 255  # 하얀색 (GRAYSCALE에서는 255가 하얀색)
    # thickness = 1  # 선 두께
    # loaded_image = cv2.line(loaded_image, start_point, end_point, color, thickness)

    loaded_image = np.flipud(loaded_image)  
    map_data = np.zeros_like(loaded_image)
    map_data[loaded_image == 255] = 1  
    map_data[loaded_image == 127] = 2  
    return map_data, loaded_image

def extract_path_coordinates(map_data):
    path_coordinates = []
    for row_idx, row in enumerate(map_data):
        for col_idx, cell in enumerate(row):
            if cell == 2:
                path_coordinates.append((col_idx,row_idx))
    return path_coordinates

def is_wall(coord, map_data):
    row, col = coord
    return not (0 <= row < map_data.shape[0] and 0 <= col < map_data.shape[1]) or map_data[row, col] == 1

def is_path_clear(coord1, coord2, map_data, threshold=22):
    if distance.cityblock(coord1, coord2) > threshold:
        return False
    c1, r1 = coord1
    c2, r2 = coord2
    num_steps = max(abs(r2 - r1), abs(c2 - c1))
    for step in range(1, num_steps + 1):
        r = int(r1 + step * (r2 - r1) / num_steps)
        c = int(c1 + step * (c2 - c1) / num_steps)
        if is_wall((r, c), map_data):
            # print(r, c)
            return False
    return True





def gaejjeonda_main(map_data, loaded_image):
    string_list = generate_string_list()
    image_path = "/home/hdk/ws/final_project/data/load_map.png"
    map_data, loaded_image = load_map(image_path)
    loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_GRAY2BGR)
    # temp_img = np.zeros((82*5,39*5,3), dtype=np.uint8)
    temp_img = cv2.resize(loaded_image, (39*5, 82*5))


    # cv2.imshow("loaded_image", loaded_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    path_coordinates = extract_path_coordinates(map_data)


    radius = 5
    # 초록색 (Green) with Alpha (투명도)
    color = (0, 255, 0)  # 마지막 128은 Alpha 값으로, 0.5 =  255 / 2 ~= 128
    alpha = 0.5

    li = []

    for p_idx1, p1 in enumerate(path_coordinates):
        center_xy = (p1[0]*5, p1[1]*5)
        cv2.putText(temp_img, string_list[p_idx1], center_xy, cv2.FONT_HERSHEY_COMPLEX, 1, (200,200,200), 2)
        cv2.circle(temp_img, center_xy, radius, color, -1)
        cnt = 0

        for p_idx2, p2 in enumerate(path_coordinates):
            if p_idx1 == p_idx2:
                continue

            if is_path_clear(p1, p2, map_data):
                start_point = (p1[0]*5, p1[1]*5)
                end_point = (p2[0]*5, p2[1]*5)
                # cv2.line(temp_img, start_point, end_point,(0,0,255), 2)
                start_point = (p1[0], p1[1])
                end_point = (p2[0], p2[1])
                # cv2.line(loaded_image, start_point, end_point,(255,0,0), 2)
                dist = math.dist(start_point, end_point)
                li.append((string_list[p_idx1], string_list[p_idx2], int(dist)))

                cnt += 1


    n = len(path_coordinates)
    nodes_list = string_list[:n]
    graph = {
        'num_nodes': n,
        'nodes': nodes_list,
        'edges': li
    }


    mst = kruskal(graph)
    # print(mst)

    for val in mst:
        # print(string_list.index(val[0]))

        p1_idx = string_list.index(val[0])
        p2_idx = string_list.index(val[1])

        p1 = path_coordinates[p1_idx]
        p2 = path_coordinates[p2_idx]

        start_point = (p1[0]*5, p1[1]*5)
        end_point = (p2[0]*5, p2[1]*5)
        # cv2.line(temp_img, start_point, end_point,(255,0,0), 2)
        start_point = (p1[0], p1[1])
        end_point = (p2[0], p2[1])
        cv2.line(loaded_image, start_point, end_point,(0,0,255), 2)

    # cv2.imshow("temp_img", temp_img)
    # cv2.imshow("map_data", map_data)
    # cv2.imshow("loaded_image", loaded_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    local_end = []
    branchs = []
    for n in nodes_list:
        cnt, _ = find_connect_nodes(mst, n)
        if cnt == 1:
            local_end.append(n)
        if cnt >= 3:
            branchs.append(n)

    test_dict = []
    for le in local_end:

        curr_node = le
        node_check_list = [False for _ in nodes_list]
        node_check_list[nodes_list.index(curr_node)] = True

        while True:
            cnt, connected_nodes = find_connect_nodes(mst, curr_node)
            if cnt == 0:
                break
            elif cnt == 1:
                curr_node = connected_nodes[0]
                node_check_list[nodes_list.index(curr_node)] = True

            elif cnt == 2:
                curr_node = connected_nodes[0]

                for cn in connected_nodes:
                    if node_check_list[nodes_list.index(cn)] == False:
                        curr_node = cn
                node_check_list[nodes_list.index(curr_node)] = True
            elif cnt >= 3:
                test_dict.append((curr_node, le))
                break

    


    curr = 'A'
    le_flag = True

    # for test_dict
    node_check_list = [False for _ in nodes_list]
    node_check_list[nodes_list.index(curr)] = True

    result_path = []
    for i in range(100):
        result_path.append(curr)
        if curr == 'X': break
        # print(le_flag)
        # print(node_check_list)
        next_branch_flag = True

        for td in test_dict:
            if le_flag:
                if td[1] == curr:
                    curr = td[0]
                    le_flag = False
                    node_check_list[nodes_list.index(curr)] = True
                    next_branch_flag = False
                    break
            else:
                if td[0] == curr:
                    if node_check_list[nodes_list.index(td[1])] == False:
                        curr = td[1]
                        le_flag = True
                        node_check_list[nodes_list.index(curr)] = True
                        next_branch_flag = False
                        break
        if next_branch_flag:
            curr = branchs[branchs.index(curr)+1]
            node_check_list[nodes_list.index(curr)] = True
    # print(result_path)

    # 그래프 생성 (딕셔너리 형태)
    graph = {}
    for u, v, w in mst:
        if u not in graph:
            graph[u] = {} # 이 부분을 추가
        graph[u][v] = w  # u -> v 간선 (가중치 w)
        if v not in graph:
            graph[v] = {} # 이 부분을 추가
        graph[v][u] = w # v -> u 간선 (가중치 w)

    path_list = []
    # print()
    for i, n in enumerate(result_path):
        if i == 0:
            continue

        
        path, _ = astar(graph, result_path[i-1], n)
        # print(f"{result_path[i-1]}에서 {n}까지의 최단 경로: {path}")
        path_list.append(path)

    return nodes_list, result_path, path_list, path_coordinates, local_end, branchs, graph


image_path = "/home/hdk/ws/final_project/data/load_map.png"
map_data, loaded_image = load_map(image_path)
nodes_list, global_path, global_path_list, path_coordinates, local_end, branchs, graph = gaejjeonda_main(map_data, loaded_image)
# # print(result_path)
# # for val in path_list:
# #     print(val)

# # print(path_coordinates)

# # # print(path_coordinates[result_path.index('X')])
# print("nodes_list")
# print(nodes_list)
# print("global_path")
# print(global_path)
# print("global_path_list")
# print(global_path_list)
# print("path_coordinates")
# print(path_coordinates)
# print("local_end")
# print(local_end)
# print("branchs")
# print(branchs)

path, _ = astar(graph, 'B', 'B')
print(f"{'B'}에서 {'B'}까지의 최단 경로: {path}")
