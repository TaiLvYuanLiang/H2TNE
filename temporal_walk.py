import sys
import os
import random
import time
import math


# fixme:直接从有效的时间窗口内的数据进行筛选
# 直接改为读取：有效窗口内的边文件input_file
def get_graph_between_time(input_file):
	graph_new = dict()
	v_list = dict()
	t_set = set()
	with open(input_file) as f:
		for line in f:
			toks = line.strip().split('\t')
			n1 = int(toks[0])
			n1_label = toks[1]
			n2 = int(toks[2])
			n2_label = toks[3]
			t = float(toks[4])
			t_set.add(t)
			v_list[n1] = n1_label
			v_list[n2] = n2_label
			# graph_new[v]保存v的满足时间条件的边
			if n1 not in graph_new:
				graph_new[n1] = dict()
			if n2 not in graph_new:
				graph_new[n2] = dict()
			if t not in graph_new[n1]:
				graph_new[n1][t] = []
			if t not in graph_new[n2]:
				graph_new[n2][t] = []
			graph_new[n1][t].append([n2, n2_label])
			graph_new[n2][t].append([n1, n1_label])
	# print(graph_new)
	return v_list, graph_new, min(t_set), max(t_set)


# 需要读取node_types文件中的边信息，即边所连接的节点的类型
def generate_node_types(node_types):
	heterg_dictionary = {}
	heterogeneous_node_types = open(node_types)

	for line in heterogeneous_node_types:
		node_type = (line.split(":")[0]).strip()
		hete_value = (line.split(":")[1]).strip()
		if node_type in heterg_dictionary.keys():
			heterg_dictionary[node_type].append(hete_value)
		else:
			heterg_dictionary[node_type] = [hete_value]
	return heterg_dictionary


class JustModel:
	def __init__(self, alpha, node_types, beta):
		self.alpha = alpha
		# self.cnt = 1
		self.homog_length = 1
		self.no_next_types = 0
		# self.heterg_probability = 0
		# self.heterg_dictionary = heterg_dictionary
		self.heterg_dictionary = generate_node_types(node_types)  # 需要读取node_types文件中的边信息，即边所连接的节点的类型

		self.beta = beta
		self.same_time_length = 1
		# self.no_next_times = 0

	# fixme：number of memorized domains = 1
	def one_step_choose_time(self, t_list, cur_time=None, isBefore=False):
		t_list.sort(reverse=isBefore)
		hit_curtime = False
		next_tid = -1  # 第一个比cur_time大的时间的下标
		for i in range(len(t_list)):
			if t_list[i] == cur_time:
				next_tid = i+1
				hit_curtime = True
				break
			if isBefore and t_list[i] < cur_time:
				next_tid = i
				break
			if (not isBefore) and t_list[i] > cur_time:
				next_tid = i
				break
		if next_tid == -1:
			return  # 没有找到符合当前时间非递减约束的时间

		furtherT_probability = 1 - math.pow(self.beta, self.same_time_length)
		r = random.uniform(0, 1)  # 随机生成0到1的数
		# print("furtherT_probability:", furtherT_probability)
		# print("r:", r)
		# exit()

		# if r <= furtherT_probability:  # 命中further概率则向前游走
		if r <= furtherT_probability or not hit_curtime:  # 命中further概率则向前游走，没有命中当前时间点也向前游走
			# 没有往后的时间片则终止游走
			if next_tid == len(t_list):
				return
			next_time = random.choice(t_list[next_tid:])
			self.same_time_length = 1
		else:
			next_time = cur_time
			self.same_time_length += 1

		# print("游走过这个时间片", next_time)
		return next_time

	# fixme：number of memorized domains = 1
	def one_step_generation(self, e_list, cur_type=None):
		if self.no_next_types == 1:
			return  # 终止循环，不找下一个节点了

		homog_type = []
		heterg_type = []
		for node_type in self.heterg_dictionary:
			if cur_type == node_type:
				homog_type = node_type
				heterg_type = self.heterg_dictionary[node_type]

		heterg_probability = 1 - math.pow(self.alpha, self.homog_length)
		r = random.uniform(0, 1)  # 随机生成0到1的数
		next_type_options = []
		if r <= heterg_probability:
			for heterg_type_iterator in heterg_type:
				next_type_options.extend([e for e in e_list if (e[1] == heterg_type_iterator)])
			if not next_type_options:
				next_type_options = [e for e in e_list if (e[1] == homog_type)]
		else:
			next_type_options = [e for e in e_list if (e[1] == homog_type)]
			if not next_type_options:
				for heterg_type_iterator in heterg_type:
					next_type_options.extend([e for e in e_list if (e[1] == heterg_type_iterator)])
		if not next_type_options:
			self.no_next_types = 1
			return  # 终止循环，不找下一个节点了

		next_node = random.choice(next_type_options)
		if next_node[1] == cur_type:
			self.homog_length = self.homog_length + 1
		else:
			self.homog_length = 1

		return next_node[0]  # next_node为向量，0号位为节点id，1号位为节点类型


#
# class JustModel_old:
# 	def __init__(self, alpha, node_types):
# 		self.alpha = alpha
# 		# self.cnt = 1
# 		self.homog_length = 1
# 		self.no_next_types = 0
# 		# self.heterg_probability = 0
# 		# self.heterg_dictionary = heterg_dictionary
# 		self.heterg_dictionary = generate_node_types(node_types)  # 需要读取node_types文件中的边信息，即边所连接的节点的类型
#
# 	# fixme：number of memorized domains = 1
# 	def one_step_generation(self, e_list, cur_type=None):
# 		if self.no_next_types == 1:
# 			return  # 终止循环，不找下一个节点了
#
# 		homog_type = []
# 		heterg_type = []
# 		for node_type in self.heterg_dictionary:
# 			if cur_type == node_type:
# 				homog_type = node_type
# 				heterg_type = self.heterg_dictionary[node_type]
#
# 		heterg_probability = 1 - math.pow(self.alpha, self.homog_length)
# 		r = random.uniform(0, 1)  # 随机生成0到1的数
# 		next_type_options = []
# 		if r <= heterg_probability:
# 			for heterg_type_iterator in heterg_type:
# 				next_type_options.extend([e for e in e_list if (e[1] == heterg_type_iterator)])
# 			if not next_type_options:
# 				next_type_options = [e for e in e_list if (e[1] == homog_type)]
# 		else:
# 			next_type_options = [e for e in e_list if (e[1] == homog_type)]
# 			if not next_type_options:
# 				for heterg_type_iterator in heterg_type:
# 					next_type_options.extend([e for e in e_list if (e[1] == heterg_type_iterator)])
# 		if not next_type_options:
# 			self.no_next_types = 1
# 			return  # 终止循环，不找下一个节点了
#
# 		next_node = random.choice(next_type_options)
# 		if next_node[1] == cur_type:
# 			self.homog_length = self.homog_length + 1
# 		else:
# 			self.homog_length = 1
#
# 		return next_node[0]  # next_node为向量，0号位为节点id，1号位为节点类型


# 基于JUST的随机游走, nodetype文件需要指定
class JustRandomWalkGenerator:
	# def __init__(self, nx_G, node_types, numwalks=10, walklength=80, p=0.2, w=2, alpha=0.2):  # alpha为停留在同一个域的概率
	def __init__(self, node_types, numwalks=10, walklength=80, p=0.2, w=2, alpha=0.2, beta=0.5):  # alpha为停留在同一个域的概率
		# self.G = nx_G
		self.just = JustModel(alpha, node_types, beta)
		# self.node_types = node_types
		# self.heterg_dictionary = generate_node_types(node_types)  # 需要读取node_types文件中的边信息，即边所连接的节点的类型
		# self.just = JustModel(alpha, self.heterg_dictionary)

		self.numwalks = numwalks  # 每个节点的walkers数
		self.walklength = walklength  # 每个walker的最长距离
		self.p = p  # 新时间片到来时，替换的新边的比例
		self.w = w  # 随机游走序列进行记录时的子窗口的大小（最多容纳的游走节点数）
		self.rm_list = dict()  # 游走序列
		# self.rm_list_label = dict()  # todo:游走序列的类型记录，just不需要记录这个
		self.rm_win_list = dict()  # todo:游走序列子窗口记录：子窗口时间戳，窗口内节点的数量
		self.change_n_list = set()  # todo:？？？

	# fixme：时间边界包含问题
	def init_walk(self, G_file, outfilename):
		# 启动时游走
		print("init walk")
		rm_list = self.rm_list
		# rm_list_label = self.rm_list_label
		rm_win_list = self.rm_win_list
		# G = self.G
		numwalks = self.numwalks
		walklength = self.walklength
		w = self.w

		# fixme：找出符合时间条件的点集、边集
		n_list, edges_new, t_old, t_new = get_graph_between_time(input_file=G_file)

		with open(outfilename, 'w'):
			pass
		# todo：考虑改用多线程进行随机游走
		# 遍历图中的每个节点 fixme：遍历当前有效的节点，同时后面要处理新旧节点问题
		# for v0 in G.nodes:	fixme: 节点先进行遍历，再重复10轮；和对每个节点都先进行重复，再遍历各个节点进行相同处理，哪种更好？
		for v0 in n_list:
			rm_list[v0] = dict()
			rm_win_list[v0] = dict()
			# 游走numwalks轮次
			for i in range(numwalks):
				rm_win_list[v0][i] = [[0.0, 0]]

				# v0是初始的节点，v是当前的节点
				v = v0
				rm = [v]
				# 游走时间初始化：指定滑动窗口最早/最晚时间；游走中t设置为游走的边的时间
				# t = t_old - 0.1
				t = t_old

				# 从v0开始一趟游走
				for l in range(walklength-1):
					# 以边类型的元路径引导游走 / 随机选取下一条要游走到的边
					# 找出符合时间条件的边集合，并选出下一跳时间time_next
					time_next = self.just.one_step_choose_time(list(edges_new[v].keys()), t)
					# 如果没有选出符合的时间，停止游走
					if not time_next:
						break

					# 在符合时间条件的边中根据just的规则选取下一条节点v_next
					# 选择的下一跳节点v_next		todo:先选时间戳还是先选类型？现在是先选时间后选类型
					v_next = self.just.one_step_generation(edges_new[v][time_next], n_list[v])

					t = time_next
					v = v_next
					rm.append(v)
					if rm_win_list[v0][i][-1][1] == w:
						rm_win_list[v0][i].append([0.0, 0])
					rm_win_list[v0][i][-1][0] = t
					rm_win_list[v0][i][-1][1] += 1

				rm_list[v0][i] = rm

			with open(outfilename, 'a') as rmf:
				for k in rm_list[v0].values():
					rmf.write(" ".join(map(str, k)) + "\n")

		self.rm_list = rm_list
		self.rm_win_list = rm_win_list
		return

	# fixme：处理新节点出现
	def back_walk(self, G_file):
		# G = self.G
		# metapath = self.metapath
		rm_list = self.rm_list
		# rm_list_label = self.rm_list_label
		rm_win_list = self.rm_win_list
		numwalks = self.numwalks
		walklength = self.walklength
		p = self.p
		w = self.w

		# 找出符合时间条件的点集、边集
		n_list, edges_new, t_old, t_new = get_graph_between_time(input_file=G_file)

		# 遍历新时间段内更新中涉及的每个节点 todo:多线程
		for v0 in n_list:
			if v0 not in rm_list:
				# todo:失效节点占用内存应当检查并清除（不影响结果）
				rm_list[v0] = dict()
				rm_win_list[v0] = dict()
				# rm_list_label[v0] = []
				for i in range(numwalks):
					rm_list[v0][i] = []
					rm_win_list[v0][i] = [[0.0, 0]]
					# rm_list_label[v0].append(0)

			# invalid_id 表示可以被用于新游走序列的id位置（过期+空位）
			invalid_ids = set()
			old_ids = set()

			# 判断这个节点上的游走序列rm_list是不是过期(或为空)
			for i in rm_list[v0]:
				if (rm_win_list[v0][i][-1][0] < t_old) or (rm_win_list[v0][i][-1][1] == 0):
					# 完全过期序列:删掉，计数
					invalid_ids.add(i)
					rm_list[v0][i] = []
					rm_win_list[v0][i] = [[0.0, 0]]
					# rm_list_label[v0][i] = 0
				else:
					old_ids.add(i)
					del_i = 0
					# 判断窗口是否过期，如果有过期的则要删去部分窗口
					for wi in range(rm_win_list[v0][i].__len__()):
						if rm_win_list[v0][i][wi][0] >= t_old:
							rm_win_list[v0][i] = rm_win_list[v0][i][wi:]
							break
						else:
							del_i += rm_win_list[v0][i][wi][1]
					rm_list[v0][i] = rm_list[v0][i][del_i:]

			# 部分过期序列计数
			old_cnt = old_ids.__len__()
			# 节点上的新边数
			new_cnt = 0
			for i in edges_new[v0]:
				new_cnt += edges_new[v0][i].__len__()

			# 把不用的序列号或者用于新序列的系列号加进 invalid_ids
			# 这里的游走轮数由这个节点上新到的边的数量和轮数上限决定
			if new_cnt+old_cnt <= numwalks:
				# 全保留，不用删
				numwalks_new = new_cnt
				# 旧序列中多余的空位怎么办？空着的部分在下一次时间到来游走中会判定为invalid序列
			elif (new_cnt > numwalks*p) and (old_cnt > numwalks-numwalks*p):
				# 都删一部分
				numwalks_new = int(numwalks*p)
				old_del_cnt = old_cnt+numwalks_new-numwalks
				temp = random.sample(old_ids, old_del_cnt)
				invalid_ids |= set(temp)
				old_ids -= invalid_ids
				for i in temp:
					rm_list[v0][i] = []
					rm_win_list[v0][i] = [[0.0, 0]]
					# rm_list_label[v0][i] = 0
			elif new_cnt <= numwalks*p:
				# 新序列全保留，旧序列删一点到 numwalks-new_cnt
				numwalks_new = new_cnt
				old_del_cnt = old_cnt+numwalks_new-numwalks
				temp = random.sample(old_ids, old_del_cnt)
				invalid_ids |= set(temp)
				old_ids -= invalid_ids
				for i in temp:
					rm_list[v0][i] = []
					rm_win_list[v0][i] = [[0.0, 0]]
					# rm_list_label[v0][i] = 0
			else:
				# 旧序列全保留，新序列删一点到 numwalks-old_cnt
				numwalks_new = numwalks-old_cnt

			# 新边逆向游走numwalks_new轮次：先经过确定
			for i in range(numwalks_new):
				use_id = invalid_ids.pop()
				# v0是初始的节点，v是当前的节点
				v = v0
				rm = [v]
				# 游走时间初始化：指定滑动窗口最早/最晚时间；游走中t设置为游走的边的时间
				# t = t_new+0.01
				t = t_new

				# 从v0开始一趟游走
				for l in range(walklength - 1):
					# 以边类型的元路径引导游走 / 随机选取下一条要游走到的边
					# 找出符合时间条件的边
					if v is None:
						break
					time_next = self.just.one_step_choose_time(list(edges_new[v].keys()), t, isBefore=True)
					# 如果没有选出符合的时间，停止游走
					if not time_next:
						break

					# 在符合时间条件的边中根据just的规则选取下一条节点v_next和确定时间戳time_next
					# 确定time_next和选择的下一跳节点v_next		todo:先选时间戳还是先选类型？现在是先选时间后选类型
					v_next = self.just.one_step_generation(edges_new[v][time_next], n_list[v])

					t = time_next
					v = v_next
					if v is None:
						break
					rm.append(v)
					if l == 0:
						rm_win_list[v0][use_id][-1] = [t, 0]
						# rm_list_label[v0][use_id] = next_type_id
					if rm_win_list[v0][use_id][-1][1] == w:
						rm_win_list[v0][use_id].append([t, 0])
					rm_win_list[v0][use_id][-1][1] += 1

				rm_list[v0][use_id] = list(rm.__reversed__())
				rm_win_list[v0][use_id] = list(rm_win_list[v0][use_id].__reversed__())

			# 剩下的旧序列运行下面的程序进行游走的补充：要指定补充的序列id、最后一次的时间、边类型是啥
			for i in old_ids:
				rm = rm_list[v0][i]
				# 游走时间初始化：序列最后的时间；游走中t设置为游走的边的时间
				t = rm_win_list[v0][i][-1][0]
				# # 元路径引导时希望游走的类型
				# next_type_id = rm_list_label[v0][i]
				# v0是初始的节点，v是当前的节点
				v = rm[-1]
				# 从序列最后一个节点开始一趟游走
				for l in range(walklength - rm.__len__()):
					# 以边类型的元路径引导游走 / 随机选取下一条要游走到的边
					# 找出符合时间条件的边
					if v is None:
						break
					time_next = self.just.one_step_choose_time(list(edges_new[v].keys()), t, isBefore=False)
					# 如果没有选出符合的时间，停止游走
					if not time_next:
						break

					# 在符合时间条件的边中根据just的规则选取下一条节点v_next和确定时间戳time_next
					# 确定time_next和选择的下一跳节点v_next		todo:先选时间戳还是先选类型？现在是先选时间后选类型
					v_next = self.just.one_step_generation(edges_new[v][time_next], n_list[v])

					t = time_next
					v = v_next
					if v is None:
						break
					rm.append(v)
					if rm_win_list[v0][i][-1][1] == w:
						rm_win_list[v0][i].append([0.0, 0])
					rm_win_list[v0][i][-1][0] = t
					rm_win_list[v0][i][-1][1] += 1

				# rm_list_label[v0][i] = next_type_id
				rm_win_list[v0][i][-1][0] = t
				rm_list[v0][i] = rm

		self.rm_list = rm_list
		self.rm_win_list = rm_win_list
		# self.rm_list_label = rm_list_label
		self.change_n_list = n_list
		return
