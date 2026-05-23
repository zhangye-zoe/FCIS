import numpy as np
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from numba import jit


@jit(nopython=True)
def check_coord(x, y, h, w):
    return x >= 0 and x < h and y >= 0 and y < w


@jit(nopython=True)
def get_graph_degree(graph):
    n = graph.shape[0]
    m = graph.shape[1]
    dirx = [0, 0, -1, -1, -1, 0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1]
    diry = [0, -1, -1, 0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 0, -1]
    offset = [0]
    degree = np.zeros((n, m), dtype=np.int16)
    for i in range(n):
        for j in range(m):
            if graph[i][j] > 0:
                for oft in offset:
                    nx = i - dirx[graph[i][j] + oft]
                    ny = j - diry[graph[i][j] + oft]
                    if (check_coord(nx, ny, n, m)):
                        degree[nx][ny] += 1
    return degree


@jit(nopython=True)
def prepare(seg, dir_graph, contour, degree):
    h, w = seg.shape[:2]
    # map between direction & coordinate offset.
    dirx = [0, 0, -1, -1, -1, 0, 1, 1, 1]
    diry = [0, -1, -1, 0, 1, 1, 1, 0, -1]

    cnt = 0
    # vis denote whether (i, j) is visited or not.
    # level demote graph depth.
    # hfa denote whether there are other points pointed to (i, j).
    vis = np.zeros((h, w), dtype=np.int16)
    level = np.zeros((h, w), dtype=np.int16)
    hfa = np.zeros((h, w), dtype=np.int16)
    level += 1
    Q = []

    for i in range(h):
        for j in range(w):
            if degree[i][j] > 0:
                seg[i][j] = 0

    for i in range(h):
        for j in range(w):
            ok1 = 0
            ok2 = 0
            if seg[i][j] == 1:
                for k in range(1, 9):
                    nx = i + dirx[k]
                    ny = j + diry[k]
                    if (nx < 0 or nx >= h or ny < 0 or ny >= w or seg[nx][ny] != 1):
                        ok1 = 1
                    if (nx < 0 and nx >= h and ny < 0 and ny >= w and contour[nx][ny] > 0):
                        ok2 = 1
            if (ok2 == 0 and ok1 == 1):
                Q.append((i, j))
                vis[i][j] = 1
            if contour[i][j] > 0 and vis[i][j] == 0:
                Q.append((i, j))
                vis[i][j] = 1
            # process point without direction.
            if dir_graph[i][j] > 0:
                nx = i + dirx[dir_graph[i][j]]
                ny = j + diry[dir_graph[i][j]]
                if (nx >= 0 and nx < h and ny >= 0 and ny < w):
                    hfa[nx][ny] = 1

    Iter = 1
    while (len(Q) > 0):
        NQ = []
        ix = 0

        Iter += 1
        cnt = 0
        while (ix < len(Q)):
            x, y = Q[ix][0], Q[ix][1]
            ix += 1
            # update the point.
            if dir_graph[x][y] != 0:
                nx = x + dirx[dir_graph[x][y]]
                ny = y + diry[dir_graph[x][y]]
                if (nx >= 0 and nx < h and ny >= 0 and ny < w and (seg[nx][ny] > 0 or seg[nx][ny] > 0)):
                    if (vis[nx][ny] == 0):
                        NQ.append((nx, ny))
                        vis[nx][ny] = Iter
                        cnt += 1
                    if (vis[nx][ny] == Iter):
                        level[nx][ny] = min(level[nx][ny], level[x][y] - 1)
                        if (dir_graph[nx][ny] == 0):
                            dir_graph[nx][ny] = dir_graph[x][y]
        # 更新4邻域的点，加入下一次迭代
        ix = 0
        while (ix < len(Q)):
            x, y = Q[ix][0], Q[ix][1]
            ix += 1
            for k in range(1, 9):
                nx = x + dirx[k]
                ny = y + diry[k]
                if (nx >= 0 and nx < h and ny >= 0 and ny < w and (seg[nx][ny] > 0) and vis[nx][ny] == 0
                        and hfa[nx][ny] == 0):
                    NQ.append((nx, ny))
                    vis[nx][ny] = Iter
                    if dir_graph[nx][ny] == 0:
                        dir_graph[nx][ny] = k
                        level[nx][ny] = min(level[nx][ny], level[x][y] - 1)
                    if level[x][y] <= -1:
                        level[nx][ny] = min(level[nx][ny], level[x][y])

        Q = NQ
    return vis, level, hfa, Q, seg


@jit(nopython=True)
def align_foreground(pred, foreground, time):
    h, w = pred.shape[:2]

    # 8 个方向对应坐标变化量
    dirx = [0, 0, -1, -1, -1, 0, 1, 1, 1]
    diry = [0, -1, -1, 0, 1, 1, 1, 0, -1]
    vis = np.zeros((h, w), dtype=np.int16)
    Q = []
    for i in range(h):
        for j in range(w):
            if pred[i][j] > 0:
                Q.append((i, j))
                vis[i][j] = 1
    Iter = 1
    while (len(Q) > 0):
        NQ = []
        ix = 0
        if Iter >= time:
            break
        Iter += 1
        while (ix < len(Q)):
            x, y = Q[ix][0], Q[ix][1]
            ix += 1
            for k in range(1, 9):
                nx = x + dirx[k]
                ny = y + diry[k]
                if (nx >= 0 and nx < h and ny >= 0 and ny < w and pred[nx][ny] == 0 and foreground[nx][ny] > 0):
                    NQ.append((nx, ny))
                    vis[nx][ny] = Iter
                    pred[nx][ny] = pred[x][y]
        Q = NQ
    return pred


def mudslide_watershed(seg, dir_graph, fore):
    seg = binary_fill_holes(seg)
    fore = binary_fill_holes(fore)
    fore = remove_small_objects(fore, 20)
    seg[fore == 0] = 0
    contour = (fore > 0) ^ (seg > 0)

    dir_graph_pos = remove_small_objects(dir_graph > 0, 20)
    dir_graph[dir_graph_pos == 0] = 0
    small_area = remove_small_objects(seg, 60) ^ seg

    du = get_graph_degree(dir_graph)
    du = du > 1
    du = remove_small_objects(du, 3)

    vis, level, hfa, Q, seg2 = prepare(seg, dir_graph, contour, du)

    thre = 0
    pred = level <= thre
    boundary = level > thre
    pred = remove_small_objects(pred, 15, connectivity=1)
    pred = pred ^ small_area

    return pred, boundary
