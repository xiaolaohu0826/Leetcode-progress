2020.02.03: 3
53, 146, 200

2020.02.04: 6
56, 76, 138, 202, 238, 1007

2020.02.05: 看到reverse linked list 想到stack 7
160, 82, 430, 203, 143, 445, 938

2020.02.09: 20
234, 141, 83, 876, 1290, 853, 88, 283, 349, 977, 167, 532, 350, 844, 278, 69, 270, 852, 155, 232

2020.02.12: 3
121, 70, 198

2020.02.13: 9
256, 746, 392, 1047(看到1D数组消消乐想到stack), 496, 937, 994(Grid 一轮一轮的update尝试用queue记录当前状态 完成一轮后push in 变化), 7, 572(尝试写一个helper比较tree，不要直接自身recurse)

2020.02.15: 6
412, 819, 811,

543(null -> 0 else max(left, right) + 1), 

176(leetcode mysql server out of date || select second/third/fourth highest: select distinct t1.salary from t1 where( select count(distinct t2.salary) from t2 where t1.salary > t2.salary) = 1/2/3) →  找出表中(大于A元素的元素 == 1)的元素,

67

2020.02.16: 7

252, 253, 33, 973,

322 solution 1: DFS + Greedy(always go with the largest coin first) + Pruning(if find remaining amount == 0, update minstep, go on, if curr step > minstep, stop):

Solution2:
DP[i] ---> least coins to make amount i
DP[i] = amount + 1, and finally check if the value changes

54(set start, endpoint corresponding to lefttop, rightbottom, and +1, -1.
倒着iterate index, 完成一行/一列加减相应的row/col:
for i in range(rightbottomrow, lefttopcol, -1) -> [rightbottomrow, lefttopcol], i--),

49

2020.02.17: 6

139(word break: DP[i]: array[0:i] can be constructed by dict? DP[0] = True -> base case, {for i in [0, len] for j in [0,i)}),

380, 240(search from top-right, if cur > target: col--, if cur < target: row++ → 如果cur > target, 那么target不可能在current column因为它是ascending),

1130(Minimum Cost Tree From Leaf Values: 元素顺序不能变 for n elements, need to multiply n-1 times and eliminate one element per time. 既然每次要消掉一个N并sum += N*另一个element，那么minimize sum的方法是使N*min(arr[indexof(N) - 1], arr[indexof(N) +1])),

91(use DP to compute 2 digits and 1 digit in once.
if 0 < s[i-1:i] <= 9, dp[i] += dp[i-1]
if 10 <= s[i-2:i] <= 26: dp[i] += dp[i-2]
so that “230” results dp: [1, 1, 2, 0] because in “30”, 0 does not add to dp[i], 30 is also out of range),

981(since timestamp inserted ascending, map<key, [valuearr, timearr]>), binary search timestamp using external library bisect)

2020.02.18: 5
31(Next Permutation: e.g. 1,2,4,6,5,3 find the element before reversely ordered subarray from the end. Then in [6,5,3], there must be an element larger than 4. Find the smallest element larger than 4 and swap. Then reverse the subarray 1,2,5, [6,4,3]to get the smallest permutation of the subarray),

17(make a dict and backtracking. Notice that if remaining string is empty, append string to result array in loop, and do not call backtracking further),

127(word change by 1 char in dict, convert beginword  to endword: use queue + BFS to loop through a turn every time, use visited to check if the neighbour is already visited),

332(Postorder DFS 强记:
visit(start):
    if start in map and !map[start].empty():
        next = map[start].pop() # 每visit完一条路就删掉它
        visit(next)
    result.append(next) # post order
do it iteratively: use stack.
# 由于是postorder traversal, reverse the result list: res[::-1]),

394(2 stacks, alphastack and numstack, insert ‘[‘ as spliter: while stack[-1] != ‘[’ … )


2020.02.21:  5

175, 177(find Nth in sql: limit, offset, group by(以便返回null)),

178(Rank Scores: 只给Scores,要求得出他们的corresponding rank: select s1.Score, (select count(distinct table2.score) from Scores s2 where table2.score >= s1.score) from Scores s1),

180(Find Consecutive Number: select distinct from l1, l2, l3 ...),

1241,

2020.02.24:  5

704, 1046(maxheap, or sort list -> bisect insort), 1209(stack[[char, freq]]),

695(number of island。
if out of bound:
  return 0
else:
  grid[i][j] = 0
  return 1 + dfs(i+1, j), dfs(i-1, j) … ),

463(island perimeter: count # of island, then for each island cell, check left and top neighbor. If neighbor: result -= 2; No need to check right and bottom because they will be counted at next stage: 一个cell的下侧如果有neighbor，说明这个neighbor上侧和cell下侧都不能算perimeter),

2020.02.25:
68(congs, finally solved a hard problem independently.
if len(word) + curlen + len(curstrlst) <= maxLineLen // len(curstrlst)是字符之间空格
    curstrlst.append
    curlen += len(word)
else:
    calculate how much space remains and how much space can be split. then find remainder, add “ “ to end of the words from left to right, one by one),

642(use trie: keep a list of children and hotlist[3]),

79(dfs just like number of island.
before dfs on neighbor, special char on self, so self cell is not used in subsequent dfs!
if len(word) == 0, return True. If index out of range or not match, return False, else continue dfs)

[212加强版 未做: return all words in matrix: 把array of words做成一个trie,
trie里有int[26]来存下一个char, 每个trie LeafNode存full string

dfs every cell, if no entry in trie, return false; else dfs],


Google OA: 
拉面题, 穷举+binary search
min distance that a node can reach farthest node in undirected graph: BFS on every node to find min.
但也可以先随便BFS一个点U 找出到所有点的距离 然后找出最远的点W UW路径的中点（可能有两个）即为下一个BFS的对象（到其他点距离最短）, O(n)

2020.02.27: 这之后未更新leetcode so far, 记得做
230(keep a global variable, subtract it in-order and check if it equals to 0),

5(longest palindrome(revisit with manacher algorithm, need a lot of review to remember this algo O(n))),

2020.02.28:
merkle tree(hash the target tree to check if any hash of subtree match the hash of target tree(e.g.leetcode 572)),

94(in-order traversal: morris traversal(O(1) space. always append current node to predecessor.right.
to get precessor:
1. predecessor = cur.left
2. while predecessor.right and predecessor.right != cur:
    predecessor = predecessor.right)
if predecessor.right: #remove the path, visit cur,cur=cur.right),

5(manacher algorithm revisited: update maxright and pos together, update maxlen and longest palindrome together),

543(revisited, compute left length and right length, check if left + right > current maxpath),

102(tree level order traversal, using stack, count level size),

107(tree level order traversal reverse: use dfs/bfs with level),

105(build tree with inorder and preorder:
def helper(startpre, startin, endin, preorder, inorder)
    preorder找root val = preorder[prestart], 在inorder里找root val对应的index i.
inorder[startin:i]为left subtree, inorder[i:endin]为right subtree. 可以用map记录{inorder[i] : i}

check if prestart >= len(preorder) or startin > start end:
    return None
root.left = f(prestart + 1, startin, i - 1, preorder, inorder)
root.right = f(prestart+i-startin+1, i+1, endin, preorder, inorder)),

144,145,

103(if count % 2 == 1: res.append(temp[::-1])),

437(use a map to record cursum and count, remember:
res += map[curr-target] + helper(left) + helper(right)
map[curr] -= 1 # this presum is only for current path!),

226(invert tree: just do left, right = right, left),

104,

257(when left and right null, append),

108(build binary tree with sorted array
        if left > right return None
        root = nums[mid],
        root.left = helper(left, mid - 1)
        root.right = helper(mid + 1, right)
        return root),

235(Lowest Common Ancestor of BST:
give nodep and nodeq,
while root:
            if p.val < root.val and q.val < root.val:
                root = root.left
            elif p.val > root.val and q.val > root.val:
                root = root.right
            else:
                return root
),

112(if not left and not right and sum == root.val: return True
        return f(left, sum - root.val) or f(right, sum - root.val)),

617(merge 2 BST, node add together:
if t1 and t2:
    build tree based on t1
else:
    return t1 or t2 # either None or all None),

110(balanced BST:
-1 means unbalanced
if abs(l - r) > 1: return -1 else return max(l, r) + 1
check the return value of helper function is -1 or not),

199(BST from right side view:
带一个list和depth = 0进去, 每层需要append一个，如果有右边append右边，如果没有就append左边
if not root:
    return
if len(res) == depth:
# 此时已添加最右值, len(res) > depth, 之后同层不动res
    res.append(root.val)
f(root.right, res, depth +1) # 右边优先check
f(root.left, res, depth +1)),

236(LCA, but tree not balanced:
check if two nodes are at the same side, if so, go further. If not, current root is the LCA because 2 nodes are separate
if not root or root == p or root == q: return None
left = …
right = …
if left and right: return root # current is LCA
return left if not right else right # go further),





