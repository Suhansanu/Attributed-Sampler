import random
import sys
import numpy as np
import timeit
import math
def algo_names(aname):
	
	def type_dict(g, attr):
		tp = dict()
		for v in g.vs:
			if str(v[attr]) == 'None' or str(v[attr]) == 'nan':
				# print v[attr]
				continue
			if v[attr] not in tp:
				tp[v[attr]] = 1
			else:
				tp[v[attr]] += 1
		return tp

	def nCr(n,r):
	    return math.factorial(n) / math.factorial(r) / math.factorial(n-r)

	def get_intial_seed(start, end, seed):
		random.seed(seed)
		return random.randint(start, end)
		# return 0
		# return np.argmax(g.closeness())

	def normalise_dict(tp):
		new_tt = dict()
		sm = sum(tp.itervalues())
		if sm > 0:
			factor = 1.0/sm
			for k in tp:
				new_tt[k] = tp[k]*factor
		else:
			return tp
		return new_tt

	def empty_dict(tp):
		for k in tp:
			tp[k] = sys.float_info.min
		return tp

	def unexplored_count(ngbs):
		count = 0
		for ngb in ngbs:
			if not ngb['frontier'] and not ngb['sampled']:
				count+=1
		return count

	def neighbour_dict(ngbs, attr):
		tp = dict()
		for ngb in ngbs:
			if ngb[attr] not in tp:
				tp[ngb[attr]] = 1
			else:
				tp[ngb[attr]] += 1
		return tp

	def unexplored_dict(ngbs, attr):
		tp = dict()
		for ngb in ngbs:
			if not ngb['frontier'] and not ngb['sampled']:
				# print ngb, attr
				if ngb[attr] not in tp:
					tp[ngb[attr]] = 1
				else:
					tp[ngb[attr]] += 1
		return tp

	def normalised_unexplored_dict(ngbs, attr):
		tp = dict()
		for ngb in ngbs:
			if not ngb['frontier'] and not ngb['sampled']:
				# print ngb, attr
				if ngb[attr] not in tp:
					tp[ngb[attr]] = 1.0 / ngb.degree()
				else:
					tp[ngb[attr]] += 1.0 / ngb.degree()
		return tp
	
	def flow_weight(a, b, attrs):
		flow = 0
		for attr in attrs:
			tp1 = dict()
			tp2 = dict()
			for ai in a:
				if ai[attr] not in tp1:
					tp1[ai[attr]] = 1
					tp2[ai[attr]] = sys.float_info.min
				else:
					tp1[ai[attr]] += 1
			for bi in b:
				if ai[attr] not in tp2:
					tp2[ai[attr]] = 1
					tp1[ai[attr]] = sys.float_info.min
				else:
					tp2[ai[attr]] += 1
			flow += stats.entropy(tp1.values(), tp2.values())
		return flow

	def maximum_pos_list(a):
		poss = []
		assert len(a) > 0, 'length less than 0 not possible!'
		mx = max(a)
		for i in range(len(a)):
			if mx == a[i]:
				poss.append(i)
		return poss[random.randint(0, len(poss)-1)]
			

	def difference_distr(old_tp, new_tp):
		try :
			total = 0.0
			if sum(old_tp.values()) == 0 or sum(new_tp.values()) == 0:
				return 1
			assert sum(old_tp.values()) >= 1-0.001  and  sum(old_tp.values()) <= 1+0.001 and  sum(new_tp.values()) >= 1-0.001  and  sum(new_tp.values()) <= 1+0.001, 'Difference in distribution problematic'+ str(old_tp.values())+ " " + str(new_tp.values())
			# if sum(old_tp.values()) >= 1-0.001 or sum(old_tp.values()) or 1+0.001 or  sum(new_tp.values()) >= 1-0.001  or  sum(new_tp.values()) <= 1+0.001:
			# 	return 1
			for key, val in old_tp.iteritems():
				if str(key) != 'None' and str(key) != 'nan':
					total+= abs(val- new_tp[key])
			return total
		except:
			return 1

	def difference_mean(old_tp, new_tp):
		total = 0.0
		e1 = 0
		e2 = 0
		for key, val in old_tp.iteritems():
			if str(key) != 'None' and str(key) != 'nan':
				e1 += val * key
				e2 += new_tp[key] * key
		return abs(e1 - e2)

	def chi_squaredp(old_tp, new_tp):
		total = 0.0
		e1 = []
		e2 = []
		for key, val in old_tp.iteritems():
			if str(key) != 'None' and str(key) != 'nan':
				if val > 0 or new_tp[key] > 0:
					e1.append(val)
					e2.append(new_tp[key])
		e2 = [sum(e1)*x for x in e2]
		# print sum(e1) , sum(e2)
		# assert int(round(sum(e1))) == 1 and int(round(sum(e2))) == 1, 'print '+ str(sum(e1)) +" " + str(sum(e2))
		# print e2, e1, chisquare(e2, e1)[1]
		# print chisquare(e2, e1)[1]
		return chisquare(e2, e1)[1]

	def merge_dict(a, b):
		c = copy.deepcopy(a)
		for key, val in b.iteritems():
			if str(key) != 'None' and str(key) != 'nan':
				if key not in c:
					c[key] = val
				else:
					c[key]+= val
		return c

	def difference_entropy(old_tp, new_tp):
		return (stats.entropy(list(old_tp.values())) - stats.entropy(list(new_tp.values())))

	def difference_kl(old_tp, new_tp):
		total = 0.0
		a = []
		b = []
		for key, val in old_tp.iteritems():
			if str(key) != 'None' and str(key) != 'nan':
				a.append(val)
				b.append(new_tp[key])
		return stats.entropy(a, b)

	def unique_coverage(new_tp):
		uniq = 0
		for key, val in new_tp.iteritems():
			if str(key) != 'None' and str(key) != 'nan':
				if val > 0.1:
					uniq+=1
		return 1.0 * uniq / len(new_tp)

	def information(p, q):
		atmp = []
		btmp = []
		if len(q) == 0:
			return 0
		# print p
		for key, val in p.iteritems():
			if val > sys.float_info.min or (key in q and q[key] > sys.float_info.min):
				atmp.append(val)
				if key not in q or q[key] <=0:
					btmp.append(sys.float_info.min)
				else:
					btmp.append(q[key])
		# print atmp, btmp
		# if str(stats.entropy(btmp, atmp)) == 'inf':
		# 	print  btmp, atmp, stats.entropy(btmp, atmp)
		return stats.entropy(btmp, atmp)

	def klinformation(p, q):
		# p is the sample frequency distribution with respect to all the attributes in the network
		# q is the new delta(v) freequecy distribution with respect to only the attributes in the delta(v)
		atmp = []
		btmp = []
		# print p, q
		for key, val in p.iteritems():
			if str(key) != 'None' and str(key) != 'nan':
				if val > sys.float_info.min or (key in q and q[key] > sys.float_info.min):
					if val > 0.1:
						atmp.append(val)
					else:
						atmp.append(sys.float_info.min)
					if key not in q or q[key] <=0:
						btmp.append(sys.float_info.min)
					else:
						btmp.append(q[key])
		
		# atmp and btmp contains the frequency distribution for all the attribute values in the sample -- ie the distinct
		# values encountered in the delta(v) is ignored in this part. This removes the 0,0 freq in both the dicts
		btmp = normalise_list(btmp)
		atmp = normalise_list(atmp)
		# print atmp, btmp
		if len(p) == 0:
			return len(btmp), 0
		bay = 0
		dst = 0
		for i in range(len(atmp)):
			# print tup
			if btmp[i] > 0:
				if atmp[i] == 0:
					dst +=1
				else:
					bay+= - math.log(btmp[i] / atmp[i]) * btmp[i]
		return 0, bay

	def reverse_information(p, q):
		atmp = []
		btmp = []
		# print p
		for key, val in p.iteritems():
			if val > 0.1 or (key in q and q[key] > 0.1):
				atmp.append(val)
				if key not in q or q[key] <=0:
					btmp.append(sys.float_info.min)
				else:
					btmp.append(q[key])
		# print atmp, btmp
		# if str(stats.entropy(btmp, atmp)) == 'inf':
		# 	print  btmp, atmp, stats.entropy(btmp, atmp)
		return stats.entropy(atmp, btmp)

	def normalise_list(a):
		if sum(a) > 0:
			return [1.0 * ai / sum(a) for ai in a]
		return a

	def bayesian(p, q):
		# p is the sample frequency distribution with respect to all the attributes in the network
		# q is the new delta(v) freequecy distribution with respect to only the attributes in the delta(v)
		atmp = []
		btmp = []
		if len(q) == 0 :
			return 0, 0
		# print p, q
		for key, val in p.iteritems():
			if str(key) != 'None' and str(key) != 'nan':
				if val > sys.float_info.min or (key in q and q[key] > sys.float_info.min):
					if val > 0.1:
						atmp.append(val)
					else:
						atmp.append(sys.float_info.min)
					if key not in q or q[key] <=0:
						btmp.append(sys.float_info.min)
					else:
						btmp.append(q[key])
		
		# atmp and btmp contains the frequency distribution for all the attribute values in the sample -- ie the distinct
		# values encountered in the delta(v) is ignored in this part. This removes the 0,0 freq in both the dicts
		btmp = normalise_list(btmp)
		atmp = normalise_list(atmp)
		# print atmp, btmp
		if len(p) == 0:
			return len(btmp), 0
		bay = 0
		dst = 0
		for i in range(len(atmp)):
			# print tup
			if btmp[i] > 0:
				if atmp[i] == 0:
					dst +=1
				else:
					bay+= - math.log(atmp[i]) * btmp[i]
		return 0, bay

	def surprise_information1(p, q):
		atmp = []
		btmp = []
		for key, val in p.iteritems():
			if str(key) != 'None' and str(key) != 'nan':
				if val > 0.1 or (key in q and q[key] > 0.1):
					atmp.append(val)
					if key not in q or q[key] <=0:
						btmp.append(sys.float_info.min)
					else:
						btmp.append(q[key])

		first  = normalise_list(btmp)
		second = normalise_list(atmp)
		return sum([-1.0 * abs(s-f) * math.log(f) for f,s in zip(first, second)])

	def surprise_information2(p, q):
		atmp = []
		btmp = []
		for key, val in p.iteritems():
			if str(key) != 'None' and str(key) != 'nan':
				if val > 0.1 or (key in q and q[key] > 0.1):
					atmp.append(val)
					if key not in q or q[key] <=0:
						btmp.append(sys.float_info.min)
					else:
						btmp.append(q[key])

		first  = normalise_list(btmp)
		second = normalise_list(atmp)
		return sum([-1.0 * (s-f) * (s-f) * math.log(f) for f,s in zip(first, second)])

	def skew_value(p, q):
		data = []
		for key, val in q.iteritems():
			data.extend([key] * int(round(val)))
		return scipy.stats.skew(data)

	def log_ks_stat(p, q):
		atmp = []
		btmp = []
		for key, val in p.iteritems():
			if str(key) != 'None' and str(key) != 'nan':
				atmp.append(val)
				if key not in q or q[key] <=0:
					btmp.append(sys.float_info.min)
				else:
					btmp.append(q[key])

		first  = normalise_list(btmp)
		second = normalise_list(atmp)
		return sum([abs(math.log(s)-math.log(f)) for f,s in zip(first, second)])

	def preferential(a):
		for i in range(len(a)):
			if a[i] < 0:
				a[i] = 0
			# assert (ai >= 0), "Preference is less than 0!!!"+str(a)

		target = random.random() * sum(a)
		pos = 0
		while target > 0:
			target-=a[pos]
			pos += 1
		# print a[pos-1], max(a)
		return pos-1


	def double_preferential(a):
		first = []
		# print a
		for ai in a:
			assert (ai[0] >= 0 and ai[1] >=0), "Preference is less than 0!!!"+str(a)
			first.append(ai[0])
		mx = max(first)
		b  = []
		for ai in a:
			if ai[0] == mx:
				b.append(ai[1])
			else:
				b.append(0)
		return preferential(b)

	def pareto_frontier(Xs, Ys, maxX = True, maxY = True):
		# Sort the list in either ascending or descending order of X
		# print Xs, Ys,
		myList = sorted([[Xs[i], Ys[i], i] for i in range(len(Xs))], reverse=maxX)
		# Start the Pareto frontier with the first value in the sorted list
		p_front = [myList[0]]    
		# Loop through the sorted list
		for pair in myList[1:]:
			if maxY:
				if pair[1] >= p_front[-1][1]: # Look for higher values of Y
					p_front.append(pair) # and add them to the Pareto frontier
			else:
				if pair[1] <= p_front[-1][1]: # Look for lower values of Y
					p_front.append(pair) # and add them to the Pareto frontier
		# Turn resulting pairs back into a list of Xs and Ys
		p_frontX = [pair[0] for pair in p_front]
		p_frontY = [pair[1] for pair in p_front]
		p_frontZ = [pair[2] for pair in p_front]
		random.shuffle(p_front)
		pos = 0
		min_dist = abs(p_front[pos][0]-p_front[pos][1])
		for jj in range(len(p_front)):
			if abs(p_front[jj][0]-p_front[jj][1]) < min_dist:
				min_dist = abs(p_front[jj][0]-p_front[jj][1])
				abs(p_front[jj][0]-p_front[jj][1])
				pos = jj

		# print p_front[pos][2]
		return p_front[pos][2]

	def gweeke(cum_new_tp, k):
		t    = len(cum_new_tp)
		ten  = int(math.floor(0.1 * t))
		fifty= int(math.floor(0.5 * t))
		last = t-1

		if len(cum_new_tp) <= 10:
			return 5, 5

		gk = []
		valids = 0.0
		for key in sorted(cum_new_tp[0][k].keys()):
			if str(key) != 'None' and str(key) != 'nan':
				s1 = []
				for i in range(ten):
					if key not in cum_new_tp[i][k]:
						s1.append(0)
					else:
						s1.append(cum_new_tp[i][k][key])

				s2 = []
				for i in range(fifty, last):
					if key not in cum_new_tp[i][k]:
						s2.append(0)
					else:
						s2.append(cum_new_tp[i][k][key])
				# print s1,s2
				# print t, '\t', np.mean(s1), '\t', np.mean(s2), '\t',  np.var(s1),  '\t', '\t',np.var(s2)
				# print key, 1.0 * abs(np.mean(s1)-np.mean(s2))/ math.sqrt(np.var(s1)+np.var(s2))
				if abs((np.mean(s1)-np.mean(s2))/ math.sqrt(np.var(s1)+np.var(s2))) <= 1:
					valids +=1
				gk.append(1.0 * abs((np.mean(s1)-np.mean(s2))/ math.sqrt(np.var(s1)+np.var(s2))))
		
		# print len(cum_new_tp), k
		return np.mean(gk), 1.0 * valids / len(cum_new_tp[0][k])

	def gelrub(cum_new_tp, k):
		# print len(cum_new_tp[0])
		last= len(cum_new_tp[0])
		if last <= 10:
			return 5

		
		# print ''
		valids = 0.0
		for key in sorted(cum_new_tp[0][0][k].keys()):
			vps = []
			mps = []
			if str(key) != 'None' and str(key) != 'nan':
				for p in range(len(cum_new_tp)):
					s1 = []
					for i in range(last):
						if key not in cum_new_tp[p][i][k]:
							s1.append(0)
						else:
							s1.append(cum_new_tp[p][i][k][key])

					# print 'here', key, p,  last, len(s1)
					mps.append(np.mean(s1))
					vps.append(np.var(s1))
			m = len(mps)
			n = last
			B = n * np.var(mps)
			W = np.mean(vps)
			Vhat = 1.0 * W*(n - 1.0)/n + 1.0* B/n
			if np.sqrt(Vhat/W) < 1.02:
				valids +=1
			
		return valids / len(cum_new_tp[0][0][k].keys())


	def re_estimated(orig_tp, std):
		new_orig_tp = dict()
		# print orig_tp
		for key, value in orig_tp.iteritems():
			if str(key) != 'None' and str(key) != 'nan':
				new_orig_tp[key] = max(0, np.random.normal(orig_tp[key], std))
		# print 'guassian', orig_tp, normalise_dict(new_orig_tp)
		return normalise_dict(new_orig_tp)


	def plot_degree_distribution(degrees, log = False):
		degs = {}
		for deg in degrees:
			if deg not in degs:
				degs[deg] = 0
			degs[deg] +=1
		items = sorted(degs.items())
		plt.clf()
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot([ k for (k , v ) in items ] , [ v for (k ,v ) in items ])

		if log:
			ax.set_xscale('log')
			ax.set_yscale('log')
		plt.title("Degree distribution")
		plt.show()
		plt.clf()


	def get_50_dataset(g, attr):
		data = []
		res  = []
		idx  = [i for i in range(len(g.vs))]
		random.shuffle(idx)
		ctr = 0
		while len(data) < 0.5 * len(g.vs):
			if g.vs[idx[ctr]]['sampled'] == 0:
				data.append([g.vs[idx[ctr]]['cont_'+str(i)] for i in range(int(attr.split('_')[1]))])
				res.append(g.vs[idx[ctr]][attr])
			# print len(data)
			ctr +=1
		return data, res

	def intialise_data(g, seed, attrs, error, pnames):
		random.seed(seed)
		props = [[] for _ in range(len(attrs))]
		for i in range(len(attrs)):
			props[i] = [[] for _ in range(5)]

		orig_d = []
		orig_c = []
		orig_star = []
		orig_cor = []
		orig_assort = []
		orig_dia = []


		# print props
		orig_tp = []
		oric_tp = []
		new_tp  = []
		total_tp= []
		cum_new_tp = []
		# print 'here'
		for attr in attrs:
			# orig_tp.append(normalise_dict(type_dict(g, attr)))
			# oric_tp.append(type_dict(g, attr))
			# new_tp.append(empty_dict(type_dict(g, attr)))
			total_tp.append(empty_dict(type_dict(g, attr)))
			# orig_star.append(_dataset.star_relation(g, attr))
			# orig_cor.append(_dataset.cor_relation(g, attr))
			# orig_assort.append(g.assortativity_nominal(attr))


		data = []
		odata= []
		centers = [[]]
		labels_true = []
		node_len = len(g.vs)
		size = round(1.0 * error* node_len/100.0)
		return props, orig_tp, oric_tp, new_tp, total_tp, node_len, size, cum_new_tp, centers, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, data, odata, labels_true

	def finalise_prop(g, props, cum_new_tp, stepsize, attrs, tmp, new_tp, orig_tp, data, odata, labels_true, centers, attr_size, datacluster_size, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, sampled_nodes, frontier, total_tp, node_len, oric_tp, weight= 1):
		for k in range(1):
			# labels_true.append(tmp['original_cluster'])
			# data.add([tmp[attr] for attr in attrs])
			# data.add((tmp[attrs[0]], tmp[attrs[1]], tmp[attrs[2]],tmp[attrs[3]]))
			# for n in tmp.neighbors():
			# 	odata.add((n[attrs[0]], n[attrs[1]], n[attrs[2]], n[attrs[3]]))
			# print tmp
			# print attr_size
			# print len(sampled_nodes)
			odata.append(tmp['continuous'])
			sampling_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
			sampling_sizes  = [ int(1.0 * sampling_point / 100 * len(g.vs)) for sampling_point in sampling_points]
			# if len(sampled_nodes) in sampling_sizes and len(sampled_nodes) > datacluster_size:
				# enc = preprocessing.OneHotEncoder(categorical_features=[i for i in range(len(attrs))])
				# enc.fit(data)
				# print odata
				# print len(sampled_nodes)
				# kmeans_model= KMeans(n_clusters=datacluster_size).fit(odata)
				# labels_predk = kmeans_model.labels_
				# st = set()
				# for d in data:
				# 	st.add((d[0], d[1], d[2]))

				# props[k][0].append(1.0*len(set(data)) / attr_size)
				# props[k][0].append(0)
				# props[k][2].append(len(odata)/len())
				# # props[k][0].append(metrics.v_measure_score(labels_true, labels_predk))
				# props[k][1].append(metrics.normalized_mutual_info_score(labels_true, labels_predk))
				# props[k][2].append(metrics.adjusted_rand_score(labels_true, labels_predk))
				# if len(set(labels_predk)) <= 1 or len(set(labels_predk)) > len(odata):
				# 	props[k][0].append(-1)
				# else:
				# 	props[k][0].append(metrics.silhouette_score(np.array(odata), np.array(labels_predk), metric='euclidean'))

				# avg_dist = np.dist()
				# props[k][1].append(np.mean( scipy.spatial.distance.cdist( g.vs['continuous'], odata) ) )
				# props[k][1].append( 0 )

				# original_labels = [vi['original_cluster'] for vi in sampled_nodes]
				# props[k][2].append(metrics.normalized_mutual_info_score(original_labels, labels_predk))

				# sigma_labels_prek = KMeans(n_clusters=datacluster_size).fit(np.array(odata) / np.std(np.array(odata), axis = 0) ).labels_
				# props[k][3].append(metrics.silhouette_score(np.array(odata) / np.std(np.array(odata), axis = 0), np.array(sigma_labels_prek), metric='euclidean'))

				# original_labels = [vi['original_cluster_sigma'] for vi in sampled_nodes]
				# props[k][4].append(metrics.normalized_mutual_info_score(original_labels, sigma_labels_prek))

				# props[k][4].append(1.0 * len(set(labels_true)) / datacluster_size)
				# print len(sampled_nodes)


				
	
	def expansion_starter(frontier, sampled_nodes, node_len, g, seed):
		if len(frontier) == 0:
			if len(sampled_nodes) == 0:
				start = g.vs[get_intial_seed(0,node_len-1, seed)]
			else:
				start = g.vs[random.randint(0,node_len-1)]
				while start['sampled']:
					start = g.vs[random.randint(0,node_len-1)]
				
			frontier.append(start)
			start['frontier'] = 1

	
	def expansion_ender(sampled_nodes, frontier, pos, new_changes, attrs, total_tp, weight = 1):
		sampled_nodes.add(frontier[pos])
		frontier[pos]['sampled'] = 1
		ngbs = set(frontier[pos].neighbors())
		# frontier_info_dict.pop(tmp, None)
		# print frontier[pos]['id']
		frontier[pos]['frontier'] = 0
		# print frontier[pos]['id'], pos
		del frontier[pos]
		new_changes.clear()
		new_tmp = []
		for ngb in ngbs:
			if not ngb['frontier'] and not ngb['sampled']:
				frontier.append(ngb)
				new_tmp.append(ngb)
			for nn in ngb.neighbors():
				if nn['frontier']:
					new_changes.add(nn['id'])
		for new in new_tmp:
			new['frontier'] = 1
			for k in range(len(attrs)):
				if str(new[attrs[k]]) != 'None' and str(new[attrs[k]]) != 'nan':
					total_tp[k][new[attrs[k]]] +=weight
		return new_changes

		






	def fast_expansion_sampling(g, seed, stepsize, attr_size, datacluster_size, attrs, error, pnames):
		props, orig_tp, oric_tp, new_tp, total_tp, node_len, size, cum_new_tp, centers, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, data, odata, labels_true = intialise_data(g, seed, attrs, error, pnames)

		g.vs['sampled']  = [0 for _ in range(node_len)]
		g.vs['frontier'] = [0 for _ in range(node_len)]
		
		sampled_nodes = set()
		frontier	  = []
		new_changes   = set()
		frontier_info_dict = dict()
		mx = 0
		while len(sampled_nodes) <= size:
			# print len(sampled_nodes), size
			expansion_starter(frontier, sampled_nodes, node_len, g, seed)
			
			#pos = findmax(frontier)
			pos = random.randint(0,len(frontier)-1)
			mx  = unexplored_count(frontier[pos].neighbors())
			for i in range(len(frontier)):
				info = 0
				if frontier[i]['id'] not in frontier_info_dict or frontier[i]['id'] in new_changes:
					# st = timeit.default_timer()
					info = unexplored_count(frontier[i].neighbors())
					frontier_info_dict[frontier[i]['id']] = info
					# tmr += timeit.default_timer() - st
				else:
					info = frontier_info_dict[frontier[i]['id']]

				# info = unexplored_count(frontier[i].neighbors())
				if info > mx:
					mx = info
					pos = i
			
			tmp = frontier[pos]
			new_changes = expansion_ender(sampled_nodes, frontier, pos, new_changes, attrs, total_tp)

			finalise_prop(g, props, cum_new_tp, stepsize, attrs, tmp, new_tp, orig_tp, data, odata, labels_true, centers, attr_size, datacluster_size, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, sampled_nodes, frontier, total_tp, node_len, oric_tp)

		

		return props, [s['id'] for s in sampled_nodes]
	
	def random_walk_sampling(g, seed, stepsize, attr_size, datacluster_size, attrs, error, pnames):
		props, orig_tp, oric_tp, new_tp, total_tp, node_len, size, cum_new_tp, centers, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, data, odata, labels_true = intialise_data(g, seed, attrs, error, pnames)

		g.vs['sampled']  = [0 for _ in range(node_len)]
		g.vs['count']  = [0 for _ in range(node_len)]
		
		sampled_nodes = set()
		t = 0
		start = current = None 
		while len(sampled_nodes) <= size:
			if len(sampled_nodes) == 0 or t > 1 * node_len:
				t = 0
				if len(sampled_nodes) == 0:
					start = g.vs[get_intial_seed(0,node_len-1, seed)]
				else:
					start = g.vs[random.randint(0,node_len-1)]
					while start['sampled']:
						start = g.vs[random.randint(0,node_len-1)]					
				current = start
			else:
				if random.random() < 0.0:
					current = start
				else:
					current = current.neighbors()[random.randint(0, len(current.neighbors())-1)]
			t+=1
			current['count']+=1
			if not current['sampled']:
				current['sampled'] = 1
				sampled_nodes.add(current)
				tmp = current
				finalise_prop(g, props, cum_new_tp, stepsize, attrs, tmp, new_tp, orig_tp, data, odata, labels_true, centers, attr_size, datacluster_size, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, sampled_nodes, [], total_tp, node_len, oric_tp, weight = 1.0 )

		# print sum(g.vs['count'])
		# plot_degree_distribution(g.vs['count'], False)

		return props, [s['id'] for s in sampled_nodes]

	def breadth_first_search_sampling(g, seed, stepsize, attr_size, datacluster_size, attrs, error, pnames):
		props, orig_tp, oric_tp, new_tp, total_tp, node_len, size, cum_new_tp, centers, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, data, odata, labels_true = intialise_data(g, seed, attrs, error, pnames)

		g.vs['visited']  = [0 for _ in range(node_len)]
		
		sampled_nodes = set()
		queue = []
		while len(sampled_nodes) <= size:
			if len(queue) == 0:
				if len(sampled_nodes) == 0:
					start = g.vs[get_intial_seed(0,node_len-1, seed)]
				else:
					start = g.vs[random.randint(0,node_len-1)]
					while start['sampled']:
						start = g.vs[random.randint(0,node_len-1)]					
				queue.append(start)
				start['visited'] = 1
			vtx = queue.pop(0)
			sampled_nodes.add(vtx)
			for n in vtx.neighbors():
				if not n['visited']:
					queue.append(n)
					n['visited'] = 1

			tmp = vtx
			finalise_prop(g, props, cum_new_tp, stepsize, attrs, tmp, new_tp, orig_tp, data, odata, labels_true, centers, attr_size, datacluster_size, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, sampled_nodes, queue, total_tp, node_len, oric_tp, weight = 1.0)

		

		return props, [s['id'] for s in sampled_nodes]

	def metropolis_hastings_random_walk_sampling(g, seed, stepsize, attr_size, datacluster_size, attrs, error, pnames):
		props, orig_tp, oric_tp, new_tp, total_tp, node_len, size, cum_new_tp, centers, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, data, odata, labels_true = intialise_data(g, seed, attrs, error, pnames)

		g.vs['sampled']  = [0 for _ in range(node_len)]
		g.vs['count']    = [0 for _ in range(node_len)]
		sampled_nodes = set()
		t = 0
		start = current = None 
		while len(sampled_nodes) <= size:
			if len(sampled_nodes) == 0 or t > 1 * node_len:
				t = 0
				if len(sampled_nodes) == 0:
					start = g.vs[get_intial_seed(0,node_len-1, seed)]
				else:
					start = g.vs[random.randint(0,node_len-1)]
					while start['sampled']:
						start = g.vs[random.randint(0,node_len-1)]					
				current = start
			else:
				if random.random() < 0.0:
					current = start
				else:
					if len(current.neighbors()) == 0:
						t = node_len
					else:
						old = current
						new = current.neighbors()[random.randint(0, len(current.neighbors())-1)]
						if random.random() < (1.0 * old.degree() / new.degree()):
							current = new
						else:
							current = old
			t+=1
			current['count']+=1
			if not current['sampled']:
				current['sampled'] = 1
				sampled_nodes.add(current)
				tmp = current
				finalise_prop(g, props, cum_new_tp, stepsize, attrs, tmp, new_tp, orig_tp, data, odata, labels_true, centers, attr_size, datacluster_size, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, sampled_nodes, [], total_tp, node_len, oric_tp, weight = 1.0)
			# print len(sampled_nodes), len([1 for v in g.vs if v['sampled'] == 1])
		# print sum(g.vs['count'])
		# plot_degree_distribution(g.vs['count'], False)
		# print props
		return props, [s['id'] for s in sampled_nodes]


	def uniform_sampling(g, seed, stepsize, attr_size, datacluster_size, attrs, error, pnames):
		props, orig_tp, oric_tp, new_tp, total_tp, node_len, size, cum_new_tp, centers, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, data, odata, labels_true = intialise_data(g, seed, attrs, error, pnames)

		g.vs['sampled']  = [0 for _ in range(node_len)]
		
		vtx = []
		for v in g.vs:
			vtx.append(v)
		random.shuffle(vtx)

		sampled_nodes = set()
		i = 0
		while len(sampled_nodes) <= size:
			vtx[i]['sampled'] = 1
			tmp = vtx[i]
			sampled_nodes.add(vtx[i])
			i+=1
			finalise_prop(g, props, cum_new_tp, stepsize, attrs, tmp, new_tp, orig_tp, data, odata, labels_true, centers, attr_size, datacluster_size, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, sampled_nodes, [], total_tp, node_len, oric_tp)

		

		return props, [s['id'] for s in sampled_nodes]

	def uniform_edge_sampling(g, seed, stepsize, attr_size, datacluster_size, attrs, error, pnames):
		props, orig_tp, oric_tp, new_tp, total_tp, node_len, size, cum_new_tp, centers, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, data, odata, labels_true = intialise_data(g, seed, attrs, error, pnames)

		g.vs['sampled']  = [0 for _ in range(node_len)]
		
		# edg = []
		# for e in g.es:
		# 	edg.append((g.vs[e.source], g.vs[e.target]))
		# random.shuffle(edg)

		sampled_nodes = set()
		ets = g.es
		vtx = g.vs
		e_len = len(g.es)-1
		while len(sampled_nodes) <= size:
			node = ets[random.randint(0, e_len)].source if random.randint(0, 1)== 0 else ets[random.randint(0, e_len)].target
			while vtx[node]['sampled'] == 1:
				node = ets[random.randint(0, e_len)].source if random.randint(0, 1)== 0 else ets[random.randint(0, e_len)].target

			vtx[node]['sampled'] = 1
			tmp = vtx[node]
			sampled_nodes.add(vtx[node])

			
			finalise_prop(g, props, cum_new_tp, stepsize, attrs, tmp, new_tp, orig_tp, data, odata, labels_true, centers, attr_size, datacluster_size, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, sampled_nodes, [], total_tp, node_len, oric_tp)
		return props, [s['id'] for s in sampled_nodes]


	def forest_fire_sampling(g, seed, stepsize, attr_size, datacluster_size, attrs, error, pnames):
		props, orig_tp, oric_tp, new_tp, total_tp, node_len, size, cum_new_tp, centers, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, data, odata, labels_true = intialise_data(g, seed, attrs, error, pnames)

		g.vs['sampled']  = [0 for _ in range(node_len)]
		g.vs['sp']  = [0 for _ in range(node_len)]
		pf = 0.7
		
		sampled_nodes = set()
		stack = []
		while len(sampled_nodes) <= size:
			if len(stack) == 0:
				if len(sampled_nodes) == 0:
					start = g.vs[get_intial_seed(0, node_len-1, seed)]
				else:
					start = g.vs[random.randint(0, node_len-1)]
				while start['sp'] == 1:
					start = g.vs[random.randint(0, node_len-1)]
				stack = [start]
			else:
				top = stack.pop()
				sampled_nodes.add(top)
				top['sampled'] = 1
				top['sp'] = 1

				ngbs = [v for v in top.neighbors()]
				random.shuffle(ngbs)
				x = np.random.geometric(pf)
				for ngb in ngbs:
					if x == 0:
						break
					if ngb['sp'] == 0:
						stack.append(ngb)
						ngb['sp'] = 1
						x -=1

				tmp = top
				finalise_prop(g, props, cum_new_tp, stepsize, attrs, tmp, new_tp, orig_tp, data, odata, labels_true, centers, attr_size, datacluster_size, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, sampled_nodes, stack, total_tp, node_len, oric_tp, weight = 1)
			

		return props, [s['id'] for s in sampled_nodes]


	def surprise_sampling(g, seed, stepsize, attr_size, datacluster_size, attrs, error, pnames):
		props, orig_tp, oric_tp, new_tp, total_tp, node_len, size, cum_new_tp, centers, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, data, odata, labels_true = intialise_data(g, seed, attrs, error, pnames)

		g.vs['sampled']  = [0 for _ in range(node_len)]
		g.vs['frontier'] = [0 for _ in range(node_len)]
		
		sampled_nodes = set()
		frontier	  = []
		new_changes = set()
		frontier_info_dict = dict()

		# seed generation
		sampled_nodes = set()
		while len(sampled_nodes) < datacluster_size:
			rnd_idx = random.randint(0, node_len-1)
			sampled_nodes.add(g.vs[rnd_idx])

		for tmp in sampled_nodes:
			sampled_nodes.add(tmp)
			tmp['sampled'] = 1
			finalise_prop(g, props, cum_new_tp, stepsize, attrs, tmp, new_tp, orig_tp, data, odata, labels_true, centers, attr_size, datacluster_size, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, sampled_nodes, frontier, total_tp, node_len, oric_tp)
		
		# frontier computation
		for seed in sampled_nodes:
			for n in seed.neighbors():
				if n['sampled'] != 1 and n['frontier'] != 1:
					frontier.append(n)
					n['frontier'] = 1

		# adding 4x well separated points
		while len(sampled_nodes) < (2 * datacluster_size):
			pos = random.randint(0, len(frontier)-1)
			tempo= np.array([frontier[pos]['cont_'+str(i)] for i in range(attr_size)])
			mx  = sum([np.linalg.norm(tempo - d) for d in odata])
			for ii in range(len(frontier)):
				tempo= np.array([frontier[ii]['cont_'+str(i)] for i in range(attr_size)])
				info  = sum([np.linalg.norm(tempo - d) for d in odata])
				if info > mx:
					mx = info
					pos= ii
			# print mx
			tmp = frontier[pos]
			new_changes = expansion_ender(sampled_nodes, frontier, pos, new_changes, attrs, total_tp)
			finalise_prop(g, props, cum_new_tp, stepsize, attrs, tmp, new_tp, orig_tp, data, odata, labels_true, centers, attr_size, datacluster_size, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, sampled_nodes, frontier, total_tp, node_len, oric_tp)
		

		# EXP many variables
		frontier_info_dict = dict()
		g.vs['d_min2'] = [np.inf for _ in range(len(g.vs))]

		means_storage, squares_storage = np.zeros(len(g.vs[0]['continuous'])), np.zeros(len(g.vs[0]['continuous']))
		for sampled_node in sampled_nodes:
			means_storage+=sampled_node['continuous']
			squares_storage+=sampled_node['continuous']**2

		while len(sampled_nodes) <= size:
			st_timer = timeit.default_timer()
			
			# print len(odata), len(sampled_nodes), mx
			root_n = math.sqrt(1.0 * len(sampled_nodes))
			nnnnnn = 1.0 * len(sampled_nodes)
			infer_sigmas = squares_storage/ nnnnnn - (means_storage/ nnnnnn)**2

			# preparing the surprise for every node in frontier and their neigbor. 
			continuous_surprise = {}
			total_set = set(frontier)
			for ii in range(len(frontier)):
				total_set|=set([v for v in frontier[ii].neighbors() if not v['sampled']])

				
			for v in total_set:
				if np.isinf(v['d_min2']):
					v['d_min2'] = np.min(np.sum((odata-np.array([v['continuous']]))**2 / infer_sigmas, axis=1))

			new_pos = random.randint(0, len(frontier)-1)
			delta_v = [frontier[new_pos]] + [v for v in frontier[new_pos].neighbors() if not v['sampled']]
			tempo_vals = np.mean([ cv['d_min2'] * nnnnnn/2.0  for cv in delta_v])
			mx = tempo_vals
			for ii in range(len(frontier)):
				delta_v = [frontier[ii]] + [v for v in frontier[ii].neighbors() if not v['sampled']]
				tempo_vals = np.mean([ cv['d_min2'] * nnnnnn/2.0  for cv in delta_v])
				info = tempo_vals
				if info > mx:
					mx = info
					new_pos= ii

			pos = new_pos
			tmp = frontier[pos]
			frontier[pos]['sampled'] = 1

			means_storage+=tmp['continuous']
			squares_storage+=tmp['continuous']**2
			for f in total_set:
				f['d_min2'] = min(f['d_min2'], np.sum( (f['continuous']-tmp['continuous'])**2 / infer_sigmas ) )

			new_changes = expansion_ender(sampled_nodes, frontier, pos, new_changes, attrs, total_tp)
			
			
			finalise_prop(g, props, cum_new_tp, stepsize, attrs, tmp, new_tp, orig_tp, data, odata, labels_true, centers, attr_size, datacluster_size, orig_d, orig_c, orig_star, orig_cor, orig_assort, orig_dia, sampled_nodes, frontier, total_tp, node_len, oric_tp)
			
			# print timeit.default_timer() - st_timer,  len(sampled_nodes), mx
		return props, [s['id'] for s in sampled_nodes]

	if aname == 'XS':
		return fast_expansion_sampling
	elif aname == 'RW':
		return random_walk_sampling
	elif aname == 'BFS':
		return breadth_first_search_sampling
	elif aname == 'MHRW':
		return metropolis_hastings_random_walk_sampling
	elif aname == 'UNI':
		return uniform_sampling
	elif aname == 'ES':
		return uniform_edge_sampling
	elif aname == 'FF':
		return forest_fire_sampling
	

	elif aname == 'SI':
		return surprise_sampling








