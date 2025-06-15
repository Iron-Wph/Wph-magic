### 二分查找
```python
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    
    while low <= high:
        mid = (low + high) // 2
        guess = arr[mid]
        
        if guess == target:
            return mid  # 找到目标值，返回索引
        elif guess > target:
            high = mid - 1  # 目标在左半部分
        else:
            low = mid + 1   # 目标在右半部分
    
    return -1  # 未找到目标值
```

### 双指针
```python
#### 在有序数组中查找两个数，使其和等于目标值
def two_sum_sorted(numbers, target):
    left, right = 0, len(numbers) - 1
    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left + 1, right + 1]  # 返回索引（题目要求从1开始）
        elif current_sum < target:
            left += 1  # 和太小，左指针右移
        else:
            right -= 1  # 和太大，右指针左移
    return []

# 示例
numbers = [2, 7, 11, 15]
target = 9
print(two_sum_sorted(numbers, target))  # 输出: [1, 2]

#### 计算长度为 k 的连续子数组的最大和
def max_sum_subarray(arr, k):
    max_sum = float('-inf')
    window_sum = 0
    window_start = 0
    
    for window_end in range(len(arr)):
        window_sum += arr[window_end]  # 扩大窗口
        if window_end >= k - 1:  # 窗口达到大小 k
            max_sum = max(max_sum, window_sum)
            window_sum -= arr[window_start]  # 缩小窗口
            window_start += 1
    return max_sum

# 示例
arr = [2, 1, 5, 1, 3, 2]
k = 3
print(max_sum_subarray(arr, k))  # 输出: 9（子数组 [5, 1, 3]）
```

### 进制转化
```python
# 十进制转其他进制
decimal = 255
print(bin(decimal))    # 输出: 0b11111111 (二进制)
print(oct(decimal))    # 输出: 0o377 (八进制)
print(hex(decimal))    # 输出: 0xff (十六进制)

# 转换回十进制
print(int('0b11111111', 2))  # 输出: 255
print(int('0o377', 8))       # 输出: 255
print(int('0xff', 16))       # 输出: 255
```

### 快速幂
![alt text](image.png)
```python
## 求 a 的 b 次幂
def binpow(a, b):
    res = 1
    while b > 0:
        if b & 1 :
            res *= a
        a = a*a
        b >>= 1
    return res 
```

### 最大公约数与最小公倍数
```python
# 最大公约数
def gcd(a, b):
    """迭代实现欧几里得算法计算最大公约数"""
    while b != 0:
        a , b = b, a % b
    return a

# 最小公倍数
abs(a * b) // gcd(a, b)
```

### 图的遍历
##### DFS
```python
DFS(v) // v 可以是图中的一个顶点，也可以是抽象的概念，如 dp 状态等。
  在 v 上打访问标记
  for u in v 的相邻节点
    if u 没有打过访问标记 then
      DFS(u)
    end
  end
end
# adj : List[List[int]] 邻接表
# vis : List[bool] 记录节点是否已经遍历


def dfs(u: int) -> None:
    vis[u] = True
    for v in adj[u]:
        if not vis[v]:
            dfs(v)

# 栈优化
def dfs_iterative(start_node):
    visited = set()
    stack = [start_node]
    
    while stack:
        node = stack.pop()  # 弹出栈顶元素
        if node not in visited:
            visited.add(node)
            print(node, end=' ')  # 处理当前节点
            # 将邻居逆序压栈，确保左子树先被访问
            for neighbor in reversed(node.neighbors):
                stack.append(neighbor)
```

##### BFS
```python
from collections import deque
def bfs(graph, start):
    """
    广度优先遍历图结构
    :param graph: 图的邻接表表示（字典形式）
    :param start: 起始节点
    :return: 遍历顺序列表
    """
    visited = set()  # 记录已访问的节点
    queue = deque([start])  # 使用双端队列
    visited.add(start)
    traversal_order = []  # 存储遍历顺序
    
    while queue:
        node = queue.popleft()  # 出队
        traversal_order.append(node)
        
        # 遍历所有邻居节点
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)  # 入队
    
    return traversal_order

# 示例用法
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print("BFS 遍历顺序:", bfs(graph, 'A'))  # 输出: ['A', 'B', 'C', 'D', 'E', 'F']
```

### 拓扑排序
```python
from collections import deque
def topological_sort_kahn(graph):
    """
    使用Kahn算法实现拓扑排序
    :param graph: 图的邻接表表示（字典形式）
    :return: 拓扑排序结果列表，如果存在环则返回空列表
    """
    # 计算每个节点的入度
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    
    # 将入度为0的节点加入队列
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        # 减少邻居节点的入度
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # 检查是否存在环
    if len(result) != len(graph):
        return []  # 存在环，无法完成拓扑排序
    
    return result

# 示例用法
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': []
}

print("Kahn算法拓扑排序结果:", topological_sort_kahn(graph))  # 输出: ['A', 'B', 'C', 'D']
```

### 最短路
##### 迪杰斯特拉
```
g = [[float'inf']]
dis = [float'inf']
vis = [False]
for i in range(n):
    dis[i ] = g[s][i]
dis[s] = 0
vis[s] = True
for i in range(0, n):
    md = float'inf'
    mi = -1
    for j in range(n):
        if not vis[j] and dis[j] < md:
            md = dis[j]
            mi = j
    if mi == -1
        return buliantong
    vis[md] = True
    for j in range(n):
        if not vis[j]:
            dis[j] = min(dis[j], md + g[mi][j])


```
```python
n, m, s = map(int, input().split())

g = [ [1e9] * n for _ in range(n)]
for _ in range(m):
    x, y, z = map(int, input().split())
    g[x-1][y-1] = z
    # g[y-1][x-1] = z
# 初始化
dis = [1e9] * n
vis = [False] * n
for i in range(n):
    dis[i] = g[s-1][i]

dis[s-1] = 0
vis[s-1] = True
for i in range(0, n):
    # 找到未访问的最小点
    md = 1e9
    mid = -1
    for j in range(n):
        if not vis[j] and dis[j] < md:
            md = dis[j]
            mid = j
    # 更新距离
    vis[mid] = True
    for j in range(n):
        if not vis[j]:
            dis[j] = min(dis[j], md + g[mid][j])

print(dis)
```

##### 佛洛依德
```python
def floyd_warshall(graph):
    """
    Floyd-Warshall算法实现所有节点对之间的最短路径
    :return: 最短距离矩阵或None（存在负权环）
    """
    nodes = list(graph.keys())
    dist = {u: {v: float('inf') for v in nodes} for u in nodes}
    
    for u in nodes:
        dist[u][u] = 0
        for v, w in graph[u].items():
            dist[u][v] = w
    
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # 检查负权环
    for node in nodes:
        if dist[node][node] < 0:
            return None  # 存在负权环
    
    return dist
```

### 最小生成树
##### Prim算法
```python
a, n = map(int, input().split())
g = [ [0] * n for _  in range(n)]
for i in range(n):
    g[i] = list(map(int, input().split()))

vis = [False] * n
dis = [1e9 ] * n

dis[0] = a
ans = 0
for i in range(n):
    mid = -1
    md = 1e9
    for j in range(n):
        if not vis[j] and dis[j] < md:
            mid = j
            md = dis[j]
    vis[mid] = True
    ans += md

    for j in range(n):
        if not vis[j] and g[mid][j] != 0:
            dis[j] = min(dis[j], g[mid][j])
print(ans)
```

##### 克鲁斯卡尔
```python
# kruskal
def find(x):
    if(x!=p[x]):
        p[x]=find(p[x])
    return p[x]

n,m=map(int,input().split())
edges=[]

for i in range(m):
    a,b,c=map(int,input().split())
    edges.append([a,b,c])
edges=sorted(edges,key=lambda x:x[2])

res=0
cnt=0
p=[i for i in range(n+1)]

for i in range(m):
    a=edges[i][0]
    b=edges[i][1]
    w=edges[i][2]
    a=find(a),b=find(b)
    if(a!=b):
        res+=w
        cnt+=1
        p[a]=b
if(cnt<n-1):
    print("impossible")
else:
    print(res)
```

### 动态规划
##### 0-1背包
![alt text](image-1.png)
```python
n, W = map(int, input().split())
w = [0] * (n + 1)
v = [0] * (n + 1)
f = [0] * (W + 1)

# 读入数据
for i in range(1, n + 1):
    w[i], v[i] = map(int, input().split())

# 动态规划处理
for i in range(1, n + 1):
    for l in range(W, w[i] - 1, -1):
        if f[l - w[i]] + v[i] > f[l]:
            f[l] = f[l - w[i]] + v[i]

# 输出结果
print(f[W])
```

##### 完全背包
![alt text](image-2.png)
```python
W, n = map(int, input().split())
w = [0] * (n + 1)
v = [0] * (n + 1)
f = [0] * (W + 1)

for i in range(1, n + 1):
    w[i], v[i] = map(int, input().split())

for i in range(1, n + 1):
    for l in range(w[i], W + 1):
        if f[l - w[i]] + v[i] > f[l]:
            f[l] = f[l - w[i]] + v[i]

print(f[W])
```

##### 最长递增子序列
```python
def lengthOfLIS(self, nums: List[int]) -> int:
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[j] + 1, dp[i])
    return max(dp)
```

##### 最长公共子序列
```python
# 两个序列长度相同
n=int(input())
a=[int(i) for i in input().split()]
b=[int(i) for i in input().split()]
f=[[0 for i in range(n+10)] for j in range(n+10)]

for i in range(n):
	for j in range(n):
		if(i>0 and j>0):
			f[i][j]=max(f[i-1][j],f[i][j-1])
		elif(i>0):
			f[i][j]=f[i-1][j]
		elif(j>0):
			f[i][j]=f[i][j-1]
		if(a[i]==b[j]):
			f[i][j]=max(f[i][j],f[i-1][j-1]+1)
res=0
for i in range(n):
	res=max(res,f[i][n-1])
print(res)
```
