import xlwt
import tensorflow as tf
from pyexcel_ods import get_data
import numpy as np
import math


def xaver_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def check_model(input_fn, purpose_fn, column_b, column_e, output_fn):

    print ('Loading data...')
    x_ods_data = get_data(input_fn, start_row=1, start_column=1)
    x_data = x_ods_data['colors_vector']

    y_ods_data = get_data(purpose_fn, start_row=1, row_limit=len(x_data),  start_column=1)
    y_data = y_ods_data['psychometricdata']

    for i in xrange(len(y_data)):
        y_data[i] = y_data[i][column_b:column_e]

    x_none = []
    y_none = []

    rows = len(x_data)
    old_cols = len(x_data[0])

    for i in xrange(rows):
        x_square = []
        x_sqrt = []
        for j in xrange(old_cols):
            x_square.append(x_data[i][j]**2)
            x_sqrt.append(math.sqrt(x_data[i][j]))
        x_data[i].extend(x_square)
        x_data[i].extend(x_sqrt)

    print (" - X_data len : ", len(x_data), len(x_data[0]))
    print (" - Y_data len : ", len(y_data), len(y_data[0]))
    print (" - W'shape is:", len(x_data[0]), '*', len(y_data[0]))
    print (" - b'shape is:", len(y_data[0]), '* 1')

    # Normalization
    print ('Normalizing...')
    new_min = 0
    new_max = 1.0
    new_range = new_max - new_min

    # Normalize the X_data
    x_old_min = []
    x_old_max = []

    for col in range(len(x_data[0])):
        old = []
        for row in range(len(x_data)):
            old.append(x_data[row][col])

        old_min = min(old)
        old_max = max(old)
        old_range = old_max - old_min

        x_old_max.append(old_max)
        x_old_min.append(old_min)

        new = [float(float(n - old_min)/float(old_range)*float(new_range)+float(new_min)) for n in old]
        for row in range(len(new)):
            x_data[row][col] = new[row]


    # Normalize the Y_data
    y_old_max = []
    y_old_min = []

    y_old_data = np.copy(y_data)
    for col in range(len(y_data[0])):
        old = []
        for row in range(len(y_data)):
            old.append(y_data[row][col])

        old_min = min(old)
        old_max = max(old)
        old_range = old_max - old_min

        y_old_max.append(old_max)
        y_old_min.append(old_min)

        new = [float(float(n - old_min)/float(old_range)*float(new_range)+float(new_min)) for n in old]
        for row in range(len(new)):
            y_data[row][col] = new[row]


    print "Configuring Model..."
    W = tf.Variable(tf.random_uniform([len(x_data[0]), len(y_data[0])], -1, 1))
    b = tf.Variable(tf.random_uniform([len(y_data[0])], -1, 1))

    temp = tf.matmul(x_data, W)
    hypothesis = tf.add(temp, b)

    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    a = tf.Variable(0.1)  # learning rate, alpha
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)  # goal is minimize cost

    # Initializing the variables
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()

    print ('Load learning coefs...')
    saver.restore(sess, 'model_bin.ckpt')

    book_out1 = xlwt.Workbook(encoding="utf-8")
    sheet_out1 = book_out1.add_sheet("weight")
    sheet_out2 = book_out1.add_sheet("b")

    ret_w = sess.run(W)
    ret_b = sess.run(b)

    len(x_data[0]), len(y_data[0])

    for i in xrange(len(x_data[0])):
        for j in xrange(len(y_data[0])):
            sheet_out1.write(i, j, round(ret_w[i][j], 8))
    for i in xrange(len(y_data[0])):
        sheet_out2.write(i, 0, round(ret_b[i], 8))

    book_out1.save("coef.xls")
    print ('Finished saving coefs successfully')

    # Testing...
    y_new_data = np.zeros((len(y_data), len(y_data[0])))
    print ('testing...')
    for i in xrange(len(y_data)):
        for j in xrange(len(y_data[0])):

            for k in xrange(len(x_data[0])):
                y_new_data[i][j] += x_data[i][k] * ret_w[k][j]

            y_new_data[i][j] += ret_b[j]

    y_new_data = np.matmul(np.array(x_data), np.array(ret_w)) + np.array(ret_b)


    # De normalizing
    y_data_denorm = np.zeros((len(y_data), len(y_data[0])))
    for col in range(len(y_data[0])):
        old = []
        for row in range(len(y_data)):
            old.append(y_new_data[row][col])

        new_min = y_old_min[col]
        new_max = y_old_max[col]
        new_range = new_max - new_min

        old_min = 0.0
        old_max = 1.0
        old_range = old_max - old_min

        new = [float(float(n - old_min)/float(old_range)*float(new_range)+float(new_min)) for n in old]
        for row in range(len(new)):
            y_data_denorm[row][col] = new[row]

    # Save the new denormalized y_data
    book_result = xlwt.Workbook(encoding="utf-8")
    sheet_out1 = book_result.add_sheet("old_y")
    sheet_out2 = book_result.add_sheet("predict_y")
    sheet_out3 = book_result.add_sheet("error")
    sheet_out4 = book_result.add_sheet("error_percent")

    error = np.zeros((len(y_data), len(y_data[0])))
    error_percent = np.zeros((len(y_data), len(y_data[0])))

    for i in xrange(len(y_data)):
        for j in xrange(len(y_data[0])):
            # Original value
            sheet_out1.write(i, j, round(y_old_data[i][j], 2))
            # Predict value
            sheet_out2.write(i, j, round(y_data_denorm[i][j], 2))

            # Error
            error[i][j] = math.fabs(float(y_old_data[i][j])-float(y_data_denorm[i][j]))
            sheet_out3.write(i, j, round(error[i][j], 2))
            # Error in percentage
            error_percent[i][j] = math.fabs(error[i][j] * 100 / (y_old_max[j] - y_old_min[j]))
            sheet_out4.write(i, j, round(error_percent[i][j], 2))

    for i in xrange(len(y_data)):
        average_error = 0.0
        max_error = 0.0
        std_dev = 0.0
        for j in xrange(len(y_data[0])):
            # find max_error
            if error[i][j] > max_error:
                max_error = error[i][j]
            average_error += error[i][j] / len(y_data[0])
            std_dev += ((error[i][j]-average_error) ** 2) / len(y_data[0])
        std_dev = math.sqrt(std_dev)
        sheet_out3.write(i, len(y_data[0]) + 1, round(average_error, 2))
        sheet_out3.write(i, len(y_data[0]) + 2, round(max_error, 2))
        sheet_out3.write(i, len(y_data[0]) + 3, round(std_dev, 2))


    for j in xrange(len(y_data[0])):
        average_error = 0.0
        max_error = 0.0
        std_dev = 0.0
        for i in xrange(len(y_data)):
            # find max_error
            if error[i][j] > max_error:
                max_error = error[i][j]
            average_error += error[i][j] / len(y_data)
            std_dev += ((error[i][j]-average_error) ** 2) / len(y_data)
        std_dev = math.sqrt(std_dev)
        sheet_out3.write(len(y_data) + 1, j, round(average_error, 2))
        sheet_out3.write(len(y_data) + 2, j, round(max_error, 2))
        sheet_out3.write(len(y_data) + 3, j, round(std_dev, 2))

        sheet_out3.write(len(y_data) + 5, j, round((average_error * 100 / (y_old_max[j] - y_old_min[j])), 2))
        sheet_out3.write(len(y_data) + 6, j, round((max_error * 100 / (y_old_max[j] - y_old_min[j])), 2))
        sheet_out3.write(len(y_data) + 7, j, round((std_dev * 100 / (y_old_max[j] - y_old_min[j])), 2))

    book_result.save(output_fn)
    print ('Write the result file.')
