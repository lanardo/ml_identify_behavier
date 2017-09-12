import tensorflow as tf
from pyexcel_ods import get_data
import math
import xlwt


def train(input_fn, purpose_fn, column_b, column_e, alpha=0.05, epochs=200001, stride=500):

    """ ---------- LOADING SESSION ------------"""

    # Load data from the training data set
    print ('Loading data...')
    print (' - x_data loading from input.ods')
    x_ods_data = get_data(input_fn, start_row=1, start_column=1)
    x_data = x_ods_data['colors_vector']
    print (' - y_data loading from purpose.ods')
    y_ods_data = get_data(purpose_fn, start_row=1, row_limit=len(x_data),  start_column=1)
    y_data = y_ods_data['psychometricdata']
    print ('Loaded successfully.')

    print ('Extracting subdata for each model...')
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

    # Save the normalizing coef
    book_out = xlwt.Workbook(encoding="utf-8")
    sheet_out1 = book_out.add_sheet("x_norm_coef")
    sheet_out2 = book_out.add_sheet("y_norm_coef")

    for i in xrange(len(x_data[0])/3):
        sheet_out1.write(0, i, x_old_max[i].__str__())
        sheet_out1.write(1, i, x_old_min[i].__str__())

        sheet_out1.write(3, i, x_old_max[i+old_cols].__str__())
        sheet_out1.write(4, i, x_old_min[i+old_cols].__str__())

        sheet_out1.write(6, i, x_old_max[i+old_cols+old_cols].__str__())
        sheet_out1.write(7, i, x_old_min[i+old_cols+old_cols].__str__())

    for i in xrange(len(y_data[0])):
        sheet_out2.write(0, i, y_old_max[i].__str__())
        sheet_out2.write(1, i, y_old_min[i].__str__())
    book_out.save("norm_coef.xls")
    print (" - Save the normalized coefficients to 'norm_coef.xls'")

    """ ---------- TRAINING SESSION ------------"""

    # Configure the model
    print ('Configuring the model...')
    W = tf.Variable(tf.random_uniform([len(x_data[0]), len(y_data[0])], -1, 1))
    b = tf.Variable(tf.random_uniform([len(y_data[0])], -1, 1))

    temp = tf.matmul(x_data, W)
    hypothesis = tf.add(temp, b)

    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    a = tf.Variable(alpha)  # learning rate, alpha
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)  # goal is minimize cost

    # Initializing the variables
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()

    print ('Load learning checkpoint...')
    saver.restore(sess, 'model_bin.ckpt')

    print ('Training...')
    for step in xrange(epochs):
        sess.run(train)
        if step % stride == 0:
            print (step, sess.run(cost))

        saver.save(sess, 'model_bin.ckpt')

    print ("Optimization Finished!")

if __name__ == '__main__':

    """
    :params
        input_fn :   input.ods       x_data
        purpose_fn : purpose.ods     y_data
        column_b :   begin column number
        column_e :   end column number
        train_step : 0.05
        limit :      200001
        step :      100
    """

    DATA_PATH = '../../../data/full/'
    RESULT_PATH = '../output/'

    train(DATA_PATH + 'inputs.ods', DATA_PATH + 'purpose_params.ods',
          column_b=5, column_e=8,
          alpha=0.05, epochs=200001, stride=100)
