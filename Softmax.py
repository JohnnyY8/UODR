#coding=utf-8
from sklearn import svm
import tensorflow as tf
import numpy as np
import random
import copy
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load all data from a cell line file
def load_data(data_file_path):
  # Load data file
  with open(data_file_path) as data_file:
    data_lines = data_file.readlines()
  # Load sample_id
  sample_id = data_lines[0].split('\t')
  sample_id[-1] = sample_id[-1][: -2]
  # Load dna_id
  dna_id = []
  for index in range(1, len(data_lines)):
    data_lines[index] = data_lines[index].split('\t')
    data_lines[index][-1] = data_lines[index][-1][: -2]
    dna_id.append(data_lines[index][0])
  # Load and transpose x_data (each row is a sample)
  x_data = []
  for col in range(1, len(sample_id) + 1):
    feature = []
    for row in range(1, len(data_lines)):
      feature.append(float(data_lines[row][col]))
    x_data.append(feature)
  return sample_id, dna_id, x_data

# Merge the samples from same medicine in this cell line
def merge_data(map_file_path, sample_id, x_data):
  # Create dict with the key of pert_id and the value of distil_id list
  res_dict = {}
  with open(map_file_path) as map_file:
    file_lines = map_file.readlines()
    for index in range(1, len(file_lines)):
      list_line = file_lines[index].split('\t')
      map_key = list_line[1]
      if not res_dict.has_key(map_key):
        res_dict[map_key] = []
      map_value = list_line[0]
      res_dict[map_key].append(map_value)
  # Find all sample from same medicine in this cell line
  list_sample_for_each_medicine = []
  list_pert = []  # All medicine in this cell line
  for key in res_dict:
    list_value = res_dict[key]
    list_temp = []
    for s_id in sample_id:
      if s_id in list_value:
        list_temp.append(s_id)
    if list_temp != []:  # Only if there are samples from the medicine in this cell line
      list_pert.append(key)
      list_sample_for_each_medicine.append(list_temp)
  # Merge the samples from same medicine
  for index_outside in range(len(list_sample_for_each_medicine)):
    old_index = []
    for s_id in list_sample_for_each_medicine[index_outside]:
      old_index.append(sample_id.index(s_id))
    old_index.sort(reverse = True)
    x_new_data_line = np.array(x_data[old_index[0]])
    del sample_id[old_index[0]]
    del x_data[old_index[0]]
    for index_inside in range(1, len(old_index)):
      x_new_data_line += np.array(x_data[old_index[index_inside]])
      del sample_id[old_index[index_inside]]
      del x_data[old_index[index_inside]]
    x_new_data_line /= len(old_index)
    sample_id.append(list_pert[index_outside])
    x_data.append(list(x_new_data_line))
  num_drug = [1 for i in range(len(sample_id))]
  return sample_id, x_data, num_drug

# Change sample_id to drug name
def change_sampleid_to_drugname(map_file_path, sample_id_2012):
  sample_id_2012_temp = copy.deepcopy(sample_id_2012)
  # Create dict with the key of pert_id and the value of distil_id list
  res_dict = {}
  with open(map_file_path) as map_file:
    file_lines = map_file.readlines()
    for index in range(1, len(file_lines)):
      list_line = file_lines[index].split('\t')
      map_key = list_line[1]
      if not res_dict.has_key(map_key):
        res_dict[map_key] = []
      map_value = list_line[0]
      res_dict[map_key].append(map_value)
  sample_id = []
  for i, iele in enumerate(sample_id_2012_temp):
    for mkey in res_dict:
      if iele in res_dict[mkey]:
        sample_id.append(mkey)
  num_drug = [0 for i in range(len(sample_id))]
  return sample_id, num_drug

# Merge_dnaid according geneid
def merge_dnaid(L1000_geneid_file_path, L1000_map_file_path, dna_id, x_data):
  dna_id_temp = copy.deepcopy(dna_id)
  x_data_temp = copy.deepcopy(x_data)
  # Load L1000 gene_id
  with open(L1000_geneid_file_path) as geneid_file:
    file_lines = geneid_file.readlines()
    del file_lines[0]
  L1000_geneid = [file_line[: -2] for file_line in file_lines]
  # Load mapping file of L1000 gene_id and dna_id
  with open(L1000_map_file_path) as L1000_map_file:
    file_lines = L1000_map_file.readlines()
    # Remove all useless lines
    for i in range(28):
      del file_lines[0]
    del file_lines[len(file_lines) - 1]
  # Construct dictionary with the key of geneid and the value of dnaid index
  dict_L1000_geneid_dnaid = {}
  list_lines = [file_line.split('\t') for file_line in file_lines]
  for list_line in list_lines:
    if list_line[3] in L1000_geneid:
      if not dict_L1000_geneid_dnaid.has_key(list_line[3]):
        dict_L1000_geneid_dnaid[list_line[3]] = [list_line[0]]
      else:
        dict_L1000_geneid_dnaid[list_line[3]].append(list_line[0])
  for each_key in dict_L1000_geneid_dnaid:
    each_value = dict_L1000_geneid_dnaid[each_key]
    list_index = []
    for value in each_value:
      list_index.append(dna_id_temp.index(value))
    list_index.sort(reverse = True)
    for each_sample in x_data_temp:
      sum_feature = 0
      for each_index in list_index:
        sum_feature += each_sample[each_index]
      new_feature = sum_feature / len(list_index)
      each_sample.append(new_feature)
    dna_id_temp.append(each_key)
  # Return the x_data and dna_id after merging
  x_data_merged_dnaid = []
  for each_sample in x_data_temp:
    x_data_merged_dnaid.append(each_sample[len(each_sample) - len(L1000_geneid): ])
  dna_id_merged = dna_id_temp[len(dna_id_temp) - len(L1000_geneid): ]
  return dna_id_merged, x_data_merged_dnaid

# Generate labels in this cell line according to drug name
def generate_label_this_cell_line_drugname(sample_id, label_file_path):
  all_label_first_letter = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
  sample_id_temp = copy.deepcopy(sample_id)
  # Load all labels
  dict_all_labels = {}
  with open(label_file_path) as label_file:
    file_lines = label_file.readlines()
    del file_lines[0]
  file_lines = [file_line.split('\t') for file_line in file_lines]
  for file_line in file_lines:
    dict_all_labels[file_line[0]] = [0 for i in range(len(all_label_first_letter))]
    if len(file_line[-1][: -2]) == 0 or file_line[2] == '':
      continue
    all_classes = file_line[-1].split(';')
    for each_classes in all_classes[: -1]:
      dict_all_labels[file_line[0]][all_label_first_letter.index(each_classes[0])] = 1
  # Generate labels in this cell line
  y_label = []
  for each_sample_id in sample_id_temp:
    if dict_all_labels.has_key(each_sample_id):
      y_label.append(dict_all_labels[each_sample_id])
    else:
      y_label.append([0 for i in range(len(all_label_first_letter))])
  return y_label

# Generate labels in this cell line
def generate_label_this_cell_line(sample_id, label_file_path, label_map_file_path):
  all_label_first_letter = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
  sample_id_temp = copy.deepcopy(sample_id)
  # Load all labels
  dict_all_labels = {}
  with open(label_file_path) as label_file:
    file_lines = label_file.readlines()
    del file_lines[0]
  file_lines = [file_line.split('\t') for file_line in file_lines]
  for file_line in file_lines:
    dict_all_labels[file_line[0]] = [0 for i in range(len(all_label_first_letter))]
    if len(file_line[-1][: -2]) == 0 or file_line[2] == '':
      continue
    all_classes = file_line[-1].split(';')
    for each_classes in all_classes[: -1]:
      dict_all_labels[file_line[0]][all_label_first_letter.index(each_classes[0])] = 1
  # Load map of pert and sample
  dict_all_samples = {}
  with open(label_map_file_path) as label_map_file:
    file_lines = label_map_file.readlines()
    del file_lines[0]
  file_lines = [file_line.split('\t') for file_line in file_lines]
  for file_line in file_lines:
    if not dict_all_samples.has_key(file_line[1]):
      dict_all_samples[file_line[1]] = [file_line[0]]
    else:
      dict_all_samples[file_line[1]].append(file_line[0])
  # Generate labels in this cell line
  y_label = []
  for each_sample_id in sample_id_temp:
    flag = 0
    for each_key in dict_all_samples:
      if each_sample_id in dict_all_samples[each_key]:
        y_label.append(dict_all_labels[each_key])
        flag = 1
        break
    if flag == 0:
      y_label.append([0 for i in range(len(all_label_first_letter))])
  return y_label

# Fileter some data that do not have label
def filter_data_and_label(sample_id, x_data, y_label, num_drug):
  sample_id_temp = copy.deepcopy(sample_id)
  x_data_temp = copy.deepcopy(x_data)
  y_label_temp = copy.deepcopy(y_label)
  num_drug_temp = copy.deepcopy(num_drug)
  filter_index = []
  sample_id_filter = []
  x_data_filter = []
  y_label_filter = []
  num_drug_filter = []
  for i, each_label in enumerate(y_label_temp):
    flag = 0
    for j in each_label:
      if j == 1.0:
        flag = 1
        break
    if flag == 0.0:
      filter_index.append(i)
  for i in range(len(sample_id_temp)):
    if i not in filter_index:
      sample_id_filter.append(sample_id_temp[i])
      x_data_filter.append(x_data_temp[i])
      y_label_filter.append(y_label_temp[i])
      num_drug_filter.append(num_drug_temp[i])
  return sample_id_filter, x_data_filter, y_label_filter, num_drug_filter

# Count the number of individual class
def count_number_class(y_label):
  sum_res = [0 for i in range(len(y_label[0]))]
  for y in y_label:
    for ind in range(len(y)):
      if y[ind] == 1:
        sum_res[ind] += 1
  return sum_res

# K fold validation
def generate_K_fold_validation(num_sample, k_fold, number_of_classes, y_label):
  y_label_temp = copy.deepcopy(y_label)
  number_of_classes_temp = copy.deepcopy(number_of_classes)
  k_test_set = []
  num_ele = num_sample / k_fold - k_fold
  limitation_each_class = [ele / k_fold for ele in number_of_classes_temp]
  al_used = []
  for i in range(k_fold):
    each_test_set = []
    count_each_class = [0 for i in range(len(y_label_temp[0]))]
    while len(each_test_set) < num_ele:
      t_num = random.randint(0, num_sample - 1)
      if t_num not in each_test_set and t_num not in al_used:
        for ind in range(len(y_label_temp[0])):
          if y_label_temp[t_num][ind] == 1 and count_each_class[ind] < limitation_each_class[ind]:
            each_test_set.append(t_num)
            al_used.append(t_num)
            count_each_class[ind] += 1
            break
    if len(k_test_set) == k_fold - 1:
      for j in range(num_sample):
        if j not in al_used:
          al_used.append(j)
          each_test_set.append(j)
    k_test_set.append(each_test_set)
  return k_test_set

# Generate the index for random batch
def generate_index(num_sample, num_batch, kth_test_set):
  kth_test_set_temp = copy.deepcopy(kth_test_set)
  batch_index = []
  while len(batch_index) < num_batch:
    t_num = random.randint(0, num_sample - 1)
    if t_num not in batch_index and t_num not in kth_test_set_temp:
      batch_index.append(t_num)
  return batch_index

# Generate the sample for random batch
def generate_batch_sample(x_data, y_label, batch_index):
  x_data_temp = copy.deepcopy(x_data)
  y_label_temp = copy.deepcopy(y_label)
  batch_xs = []
  batch_ys = []
  for index in batch_index:
    batch_xs.append(x_data_temp[index])
    batch_ys.append(y_label_temp[index])
  batch_xs = np.array(batch_xs, dtype = "float32")
  batch_ys = np.array(batch_ys, dtype = "float32")
  return batch_xs, batch_ys

# Count 01 change
def count_01_change(prob, batch_ys, threshold):
  prob_temp = copy.deepcopy(np.array(prob))
  batch_ys_temp = copy.deepcopy(batch_ys)
  prob_temp[np.where(prob_temp >= threshold)] = 1
  prob_temp[np.where(prob_temp < threshold)] = 0
  pred_t_each_class = [0.0 for i in range(len(batch_ys[0]))]
  count_each_class = [0.0 for i in range(len(batch_ys[0]))]
  pred_f = 0.0
  pred_f_u = 0.0
  pred_t = 0.0
  pred_t_u = 0.0
  pred_t_k = 0.0
  for i, iele in enumerate(prob_temp):
    flag = 0
    for j, jele in enumerate(iele):
      if batch_ys_temp[i][j] == 1:
        count_each_class[j] += 1
      if batch_ys_temp[i][j] == 1 and jele == 1:
        pred_t += 1
        pred_t_each_class[j] += 1
        for k, kele in enumerate(iele):
          if batch_ys_temp[i][k] == 0 and kele == 1:
            pred_t_u += 1
            flag = 1
            break
        if flag == 0:
          flag = 1
          pred_t_k += 1
        break
    if flag == 0:
      pred_f += 1
      for j, jele in enumerate(iele):
        if batch_ys_temp[i][j] == 0 and jele == 1:
          pred_f_u += 1
  return pred_f, pred_f_u, pred_t, pred_t_u, pred_t_k, np.array(pred_t_each_class) / np.array(count_each_class)

# Count right for each sample in validation 
def update_right_count(prob, batch_ys, right_count):
  for i, iele in enumerate(prob):
    for j, jele in enumerate(iele):
      if batch_ys[i][j] == 1 and jele == 1:
        right_count[i] += 1
        break

# Write high prob samples to file
def write_high_prob_sample_to_file(right_count, kth_test_set, smaple_id):
  right_count_temp = copy.deepcopy(right_count)
  kth_test_set_temp = copy.deepcopy(kth_test_set)
  sample_id_temp = copy.deepcopy(sample_id)
  #file_right_count_sample = open("./file_right_count_sample.txt", 'a')
  file_right_count_sample = open("./rn_L1000_cp_labeled_MCF7.txt", 'a')
  print "The right count for each test sample:", right_count_temp
  right_count_threshold = raw_input("Input the right_count_threshold: ")
  #right_count_threshold = 40
  line_str = ""
  for i, iele in enumerate(right_count_temp):
    if iele >= int(right_count_threshold):
      line_str += sample_id_temp[kth_test_set_temp[i]] + '\n'
  file_right_count_sample.write(line_str)
  file_right_count_sample.write("-----------------------------------end-----------------------------------------\n")
  file_right_count_sample.close()

# Generating strong samples for k_test_set
def generate_strong_samples(strong_file_path, sample_id):
  sample_id_temp = copy.deepcopy(sample_id)
  k_test_set = []
  with open(strong_file_path) as strong_file:
    file_lines = strong_file.readlines()
  for file_line in file_lines:
    if file_line.startswith("----"):
      continue
    index = sample_id_temp.index(file_line[: -1])
    if index not in k_test_set:
      k_test_set.append(index)
  return [k_test_set]

# Generating all indices of unkown samples
def generate_u_index(prob, batch_ys, test_threshold, kth_test_set):
  prob_temp = copy.deepcopy(prob)
  batch_ys_temp = copy.deepcopy(batch_ys)
  kth_test_set_temp = copy.deepcopy(kth_test_set)
  prob_temp[np.where(prob_temp >= test_threshold)] = 1
  prob_temp[np.where(prob_temp < test_threshold)] = 0
  pred_t_u_index = []
  pred_f_u_index = []
  for i, iele in enumerate(prob_temp):
    flag = 0
    for j, jele in enumerate(iele):
      if batch_ys_temp[i][j] == 1 and jele == 1:
        flag = 1
        for k, kele in enumerate(iele):
          if batch_ys_temp[i][k] == 0 and kele == 1:
            print "insert into pred_t_u_index", batch_ys_temp[i][k], kele
            pred_t_u_index.append(kth_test_set_temp[i])
            break
        break
    if flag == 0:
      for j, jele in enumerate(iele):
        if batch_ys_temp[i][j] == 0 and jele == 1:
          print "insert into pred_f_u_index", batch_ys_temp[i][j], jele
          pred_f_u_index.append(kth_test_set_temp[i])
          break
  return pred_t_u_index, pred_f_u_index

# Validata the unknown samples
def val_unknown_sample(prob, batch_ys, sample_id, kth_test_set, test_threshold, it_num, num_drug):
  prob_temp = copy.deepcopy(prob)
  batch_ys_temp = copy.deepcopy(batch_ys)
  sample_id_temp = copy.deepcopy(sample_id)
  kth_test_set_temp = copy.deepcopy(kth_test_set)
  num_drug_temp = copy.deepcopy(num_drug)
  prob_temp[np.where(prob_temp >= test_threshold)] = 1
  prob_temp[np.where(prob_temp < test_threshold)] = 0
  with open("./val_unknown_samples.txt", 'a') as file_p:
    for i, iele in enumerate(kth_test_set_temp):
      str_line = sample_id_temp[iele] + '\t' + str(num_drug_temp[iele]) + '\t'
      for j in batch_ys_temp[i]:
        str_line += str(j) + '\t'
      for j in prob_temp[i]:
        str_line += str(j) + '\t'
      str_line += '\n'
      file_p.write(str_line)
    file_p.write("--------------------Iteration:" + str(it_num) + "-----------------------\n")  
    if it_num == 199:
      file_p.write("--------------------END-----------------------\n")

# Build graph and start training
def start_building_and_training(num_sample, num_batch, sample_id, x_data, dna_id, y_label, k_test_set, num_drug, number_of_classes):
  sample_id_temp = copy.deepcopy(sample_id)
  x_data_temp = copy.deepcopy(x_data)
  dna_id_temp = copy.deepcopy(dna_id)
  y_label_temp = copy.deepcopy(y_label)
  k_test_set_temp = copy.deepcopy(k_test_set)
  num_drug_temp = copy.deepcopy(num_drug)
  # Build the graph
  with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, [None, 978])
    W = tf.Variable(tf.zeros([978, 14]))
    b = tf.Variable(tf.zeros([14]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_reg = tf.maximum(y, 1e-30)
    y_ = tf.placeholder(tf.float32, [None, 14])
    lambda_reg = 1
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_reg), reduction_indices=[1])) + lambda_reg * l2_loss
    train_step = tf.train.GradientDescentOptimizer(0.06).minimize(cross_entropy)
  train_threshold = 0.06
  test_threshold = 0.06
  num_batch = 100
  print "Building the graph is done..."
  # Launch the graph
  init = tf.initialize_all_variables()
  sess = tf.Session()
  file_p = open("softmax1.txt", 'w')
  for k in range(len(k_test_set_temp)):
    sess.run(init)
    for i in range(1000):
      #if i % 100 == 0:
      if i < 1000:
        batch_index = generate_index(num_sample, num_batch, k_test_set[k])
        batch_xs, batch_ys = generate_batch_sample(x_data_temp, y_label_temp, batch_index)
        prob = sess.run(y, feed_dict = {x: batch_xs, y_: batch_ys})
        pred_f, pred_f_u, pred_t, pred_t_u, pred_t_k, pred_t_each_class = count_01_change(prob, batch_ys, train_threshold)
        train_ac = pred_t / num_batch
        print "There are %d traning samples:" % (num_batch)
        print "  The training accuracy is %.2f." % (pred_t / num_batch)
        print "  The training error rate is %.2f." % (pred_f / num_batch)
        batch_xs, batch_ys = generate_batch_sample(x_data_temp, y_label_temp, k_test_set_temp[k])
        prob = sess.run(y, feed_dict = {x: batch_xs, y_: batch_ys})
        pred_f, pred_f_u, pred_t, pred_t_u, pred_t_k, pred_t_each_class = count_01_change(prob, batch_ys, test_threshold)
        val_accu = pred_t / len(k_test_set_temp[k])
        print "There are %d test samples, in %dth fold and step %d:" % (len(k_test_set_temp[k]), k, i)
        print "  The test accuracy is %.2f." % (pred_t / len(k_test_set_temp[k]))
        print "  The test error rate is %.2f." % (pred_f / len(k_test_set_temp[k]))
        print "  There are %d unpredicted samples in true positive." % (pred_t_u)
        if 99 < i < 200:
          val_unknown_sample(prob, batch_ys, sample_id_temp, k_test_set[k], test_threshold, i, num_drug)
        #print "  The accuracy for each class:", pred_t_each_class
        print "-----------------------------------end-----------------------------------------"
      batch_index = generate_index(num_sample, num_batch, k_test_set[k])
      batch_xs, batch_ys = generate_batch_sample(x_data_temp, y_label_temp, batch_index)
      loss_val = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
      file_p.write(str(train_ac) + '\t' + str(val_accu) + '\t' + str(loss_val) + '\n')
    file_p.write("----------------------end--------------------------")
  file_p.close()

# Train CNN
def train_CNN(num_sample, num_batch, sample_id, x_data, dna_id, y_label, k_test_set):
  sample_id_temp = copy.deepcopy(sample_id)
  x_data_temp = copy.deepcopy(x_data)
  dna_id_temp = copy.deepcopy(dna_id)
  y_label_temp = copy.deepcopy(y_label)
  k_test_set_temp = copy.deepcopy(k_test_set)

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(x, W):
    # Given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_1x4(x):
    # strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='SAME')

  with tf.device('/gpu:0'):
    # The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels.
    x = tf.placeholder(tf.float32, [None, 978])
    x_input = tf.reshape(x, (-1, 1, 978, 1))
    W_conv1 = weight_variable([1, 25, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_input, W_conv1) + b_conv1)
    h_pool1 = max_pool_1x4(h_conv1)

    W_conv2 = weight_variable([1, 25, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_1x4(h_conv2)

    W_fc1 = weight_variable([1 * 62 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 62 *64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 14])
    b_fc2 = bias_variable([14])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y_conv_reg = tf.maximum(y_conv, 1e-30)
    y_ = tf.placeholder(tf.float32, [None, 14])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv_reg), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy)
    print "Building CNN is done..."

  # Launch the graph
  training_threshold = 0.3
  testing_threshold = 0.06
  num_batch = 100
  init = tf.initialize_all_variables()
  sess = tf.Session()
  file_p = open("./zzt_CNN.txt", 'w')
  for k in range(len(k_test_set_temp)):
    sess.run(init)
    for i in range(500):
      #if i % 100 == 0:
      if i < 500:
        batch_index = generate_index(num_sample, num_batch, k_test_set[k])
        batch_xs, batch_ys = generate_batch_sample(x_data_temp, y_label_temp, batch_index)
        prob = sess.run(y_conv, feed_dict = {x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        pred_f, pred_f_u, pred_t, pred_t_u, pred_t_k, pred_t_each_class = count_01_change(prob, batch_ys, testing_threshold)
        train_ac = pred_t / num_batch
        print "There are %d traning samples:" % (num_batch)
        print "  The training accuracy is %.2f." % (pred_t / num_batch)
        print "  The error rate is %.2f" % (pred_f / num_batch)
        batch_xs, batch_ys = generate_batch_sample(x_data_temp, y_label_temp, k_test_set_temp[k])
        prob = sess.run(y_conv, feed_dict = {x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        pred_f, pred_f_u, pred_t, pred_t_u, pred_t_k, pred_t_each_class = count_01_change(prob, batch_ys, training_threshold)
        val_accu = pred_t / len(k_test_set_temp[k])
        print "There are %d test samples, in %dth fold and step %d:" % (len(k_test_set_temp[k]), k, i)
        print "  The test accuracy is %.2f." % (pred_t / len(k_test_set_temp[k]))
        print "  The error rate is %.2f." % (pred_f / len(k_test_set_temp[k]))
        print "  There are %d unpredicted samples in true positive." % (pred_t_u)
        print "-----------------------------------end-----------------------------------------"
      batch_index = generate_index(num_sample, num_batch, k_test_set[k])
      batch_xs, batch_ys = generate_batch_sample(x_data_temp, y_label_temp, batch_index)
      loss_val = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
      str_line = str(train_ac) + '\t' + str(val_accu) + '\t' + str(loss_val) + '\n'
      file_p.write(str_line)
    file_p.write("-----------------------------------end-----------------------------------------")
  file_p.close()

# SVM training
def SVM_training_and_testing(x_data_merged_dnaid_seperate, y_label_seperate, k_test_set):
  # Construct traing and testing set
  for k in range(len(k_test_set)):
    x_data_train = []
    y_label_train = []
    x_data_test = []
    y_label_test = []
    for i in range(len(x_data_merged_dnaid_seperate)):
      if i not in k_test_set[k]:
        x_data_train.append(x_data_merged_dnaid_seperate[i])
        y_label_train.append(y_label_seperate[i])
      else:
        x_data_test.append(x_data_merged_dnaid_seperate[i])
        y_label_test.append(y_label_seperate[i])
    # Construct 14 SVM and corresponding label list for training
    list_cls = [svm.SVC() for i in range(len(y_label_train[0]))]
    for c_ind in range(len(y_label_train[0])):
      list_y_label = [0 for i in range(len(y_label_train))]
      for index, each_y_label in enumerate(y_label_train):
        if each_y_label[c_ind] == 1:
          list_y_label[index] = 1
      list_cls[c_ind].fit(x_data_train, list_y_label)
    print "SVM training is done..."
    # SVM testing
    ndarray_res = [[] for i in range(len(x_data_test))]
    for c_ind in range(len(list_cls)):
      pred_res = list_cls[c_ind].predict(x_data_test)
      for i, ele in enumerate(pred_res):
        ndarray_res[i].append(ele)
    # Evaluating
    pred_f, pred_f_u, pred_t, pred_t_u, pred_t_k, pred_t_each_class = count_01_change(ndarray_res, y_label_test, 0.9)
    #number_of_class = Count_number_class(y_label_test)
    print "There are %d test samples, the results in %dth fold:" % (len(k_test_set[k]), k)
    print "预测错了%d个样本，错误率为%f:" % (pred_f, pred_f / len(k_test_set[k]))
    print "预测对了%d个样本，正确率为%f:" % (pred_t, pred_t / len(k_test_set[k]))
    print "预测对的样本里，有%d个样本出现了新标签:" % (pred_t_u)
    #print "The test accuracy for each class is:", np.array(pred_t_each_class) / np.array(number_of_class)

# Random Forest training
def random_forest(x_data, y_label, k_test_set):
  x_data = np.array(x_data)
  y_label = np.array(y_label)
  feature_test = x_data[k_test_set[0]]
  target_test = y_label[k_test_set[0]]
  feature_train = np.array([ele for index, ele in enumerate(x_data) if index not in k_test_set[0]])
  target_train = np.array([ele for index, ele in enumerate(y_label) if index not in k_test_set[0]])
  print "Random Forest training starting..."
  results = ""
  for number_of_estimators in range(50,1050,50):
    #分类型决策树
    clf = RandomForestClassifier(n_estimators = number_of_estimators)
    #训练模型
    s = clf.fit(feature_train, target_train)
    #评估模型准确率
    mean_r = clf.score(feature_test, target_test)
    results = results+'when n_estimators = '+str(number_of_estimators)+': mean_r = '+str(mean_r)
    #for number_class in range(0,14,):
    #    index_set = [i for i, ele in enumerate(target_test) if ele[number_class] == 1]
    #    r = clf.score(feature_test[index_set], target_test[index_set])
    #    results = results + ', '+str(number_class)+' = '+ str(r)
    results = results+'\r'
    print results
    print '---------------------------------end-----------------------------------'

if __name__ == "__main__":
  # File path
  data_root_path = "./"
  all_root_path = "./"

  # Load data
  data_file_path = os.path.join(data_root_path, "L1000_cp_labeled_PC3.txt")
  sample_id, dna_id, x_data = load_data(data_file_path)
  sample_id_2012, dna_id_2012, x_data_2012 = load_data(data_file_path)
  print "Loading data is done..."

  # Merge data from the same drug
  map_file_path = os.path.join(all_root_path, "trt_cp.info")
  sample_id, x_data, num_drug = merge_data(map_file_path, sample_id, x_data)
  print "Merging data is done..."

  # Merge L1000 dna_id according to gene_id
  L1000_geneid_file_path = os.path.join(all_root_path, "Gene_ID.txt")
  L1000_map_file_path = os.path.join(all_root_path, "GPL96.annot")
  dna_id, x_data = merge_dnaid(L1000_geneid_file_path, L1000_map_file_path, dna_id, x_data)
  dna_id_2012, x_data_2012 = merge_dnaid(L1000_geneid_file_path, L1000_map_file_path, dna_id_2012, x_data_2012)
  print "Merging L1000 dna_id according to gene_id is done..."

  # Generate y_label in this cell line
  label_file_path = os.path.join(all_root_path, "L1000_smallcompound_labeled_detail.info")
  y_label = generate_label_this_cell_line_drugname(sample_id, label_file_path)
  label_map_file_path = os.path.join(all_root_path, "selected_sample_2012.txt")
  y_label_2012 = generate_label_this_cell_line(sample_id_2012, label_file_path, label_map_file_path)
  print "Generating label in this cell line is done..."

  # Change sample_id_2012 into drug name
  sample_id_2012, num_drug_2012 = change_sampleid_to_drugname(map_file_path, sample_id_2012)
  print "Changing sample_id into drug name is done..."

  # Merge data and labels from same drug and 2012
  x_data += x_data_2012
  sample_id += sample_id_2012
  y_label += y_label_2012
  num_drug += num_drug_2012
  print "Mergeing data and labels from same drug and 2012 is done..."

  # Filter x_data_merged_dnaid and y_label
  sample_id, x_data, y_label, num_drug = filter_data_and_label(sample_id, x_data, y_label, num_drug)
  print "Filtering data and label is done..."

  # Count number of individual class
  number_of_classes = count_number_class(y_label)
  print "Counting number of classes is done..."

  # K fold validation
  k_fold = 10
  num_sample = len(sample_id)
  k_test_set = generate_K_fold_validation(num_sample, k_fold, number_of_classes, y_label)
  print "K fold validation is done..."

  # Build graph and start training
  num_batch = 100
  num_sample = len(sample_id)
  start_building_and_training(num_sample, num_batch, sample_id, x_data, dna_id, y_label, k_test_set, num_drug, number_of_classes)
  print "Classification is done..."
