from config import NUM_TRAINING_ITERATIONS, CONVERGENCE_THRESHOLD
from formulas import err
from models import Layer, cfile

f = None
curr_point = 0
target = []
attrs = []
total_runs = 0
data = None
num_incorrect = 0
prev_sample_err = 0
curr_sample_err = 0

def parse_data(fname):
    # reset all the data
    global curr_point
    global total_runs
    global target
    global attrs
    global num_incorrect
    global prev_sample_err
    global curr_sample_err
    global data
    global f

    curr_point = 0
    total_runs = 0
    target = []
    attrs = []
    num_incorrect = 0
    prev_sample_err = 0
    curr_sample_err = 0
    

    # set the proper data file
    data_file = 'error.txt'
    if fname == 'training_data.txt':
        data_file = 'training_error.txt'
    elif fname == 'validation_data.txt':
        data_file = 'validation_error.txt'
    elif fname == 'testing_data.txt':
        data_file = 'testing_error.txt'


    # clear the file
    open(data_file, 'w+').close()

    # open the data file for logging
    data = cfile(data_file, 'w')

    f = open(fname, 'r').readlines()

    for row in f:
        row = [x.strip() for x in row.split(',')]
        row = [int(num) for num in row]    
        target.append(int(row[0])) 
        attrs.append(row[1:])

if __name__ == '__main__':
    print ("analyzing the training dataset...")
    # analyze the training dataset and store its information into globals
    parse_data('training_data.txt')

    # Layer function with nodes and layer number    
    x = Layer(10, attrs[curr_point], 1)
    y = Layer(5, x.layer_out, 2)


    print ("Intialization of training the neural network:")
    # iterating to find point of convergence for training data
    while total_runs < NUM_TRAINING_ITERATIONS:   
        # setting new input value
        x.input_vals = attrs[curr_point]

        # first layer and it's evaluation
        x.input_vals = attrs[curr_point]
        x.eval()

        # second layer and it's evaluation
        y.input_vals = x.layer_out
        y.eval()

        # backpropogation is used
        y.backprop(target[curr_point])        
        x.backprop(y)

        # calcualtion of current error
        curr_err = err(y.layer_out[0], target[curr_point])

        # round up and down for e and p label
        if y.layer_out[0] >= 0.5:
            temp = 1
        else:
            temp = 0

        # number of wrong target increamented
        if(temp != target[curr_point]):
            num_incorrect += 1

        # checking for convergence
        if total_runs % 100 == 0:
            prev_sample_err = curr_sample_err
            curr_sample_err = curr_err
            if abs(prev_sample_err - curr_sample_err) < CONVERGENCE_THRESHOLD:
                print ("Given training data is converged at " + str(total_runs) + "iteration")
                break;

        # current itration information
        print ("ongoing iteration: " + str(total_runs))
        print ("ongoing error: " + str(curr_err) + "\n")
        data.w(curr_err)

        # incrementing total runs and current point
        total_runs += 1
        curr_point += 1

        if curr_point >= len(f):
            curr_point = 0

    # close the file
    data.close()



    print ("training is done now hit enter button to start validation.")
    print ("percentage error on training dataset: " + str(float(num_incorrect)/NUM_TRAINING_ITERATIONS))
    #raw_input()
    input()


    print ("analyzing the validation dataset...")
    # analyzing of the validation dataset 
    parse_data('validation_data.txt')

    

    print ("Initialization of validating the neural network:")
    # iterating to find point of convergence for validation data
    while total_runs < len(f):
                             
        # setting new input values
        x.input_vals = attrs[curr_point]

        # first layer and it's evaluation
        x.input_vals = attrs[curr_point]
        x.eval()

        # second layer and it's evaluation
        y.input_vals = x.layer_out
        y.eval()

        # backpropogation is used
        y.backprop(target[curr_point])
        x.backprop(y)

        # calculating current error
        curr_err = err(y.layer_out[0], target[curr_point])

        # round up and down for e and p label
        if y.layer_out[0] >= 0.5:
            temp = 1
        else:
            temp = 0

        # number of wrong target incremented
        if(temp != target[curr_point]):
            num_incorrect += 1

        # checking for convergence
        if total_runs % 100 == 0:
            prev_sample_err = curr_sample_err
            curr_sample_err = curr_err
            if abs(prev_sample_err - curr_sample_err) < CONVERGENCE_THRESHOLD:
                print ("validation data is converged at " + str(total_runs) + "iteration.")
                break;

        #current iteration information
        print ("Ongoing iteration: " + str(total_runs))
        print ("Ongoing error: " + str(curr_err) + "\n")
        data.w(curr_err)

        # incrementing total runs and current point
        total_runs += 1
        curr_point += 1

        if curr_point >= len(f):
            curr_point = 0

    # close the file
    data.close()



    print ("validation part is done now hit enter button to test it.")
    print ("Percentage error on validation set: " + str(float(num_incorrect)/NUM_TRAINING_ITERATIONS))
    #raw_input()
    input()

    print ("Intialization of testing the neural network:")
    # Analyzing the testing data 
    parse_data('testing_data.txt')

    # iterating to find point of convergence for testing data
    while curr_point < len(f):
        
        # setting new input values
        x.input_vals = attrs[curr_point]
                             
        # first layer and it's evaluation
        x.input_vals = attrs[curr_point]
        x.eval()
                             
        # second layer and it's evaluation
        y.input_vals = x.layer_out
        y.eval()

        # calculation of current error
        curr_err = err(y.layer_out[0], target[curr_point])

        # round up and down for e and p label
        if y.layer_out[0] >= 0.5:
            temp = 1
        else:
            temp = 0

        # number of wrong target incremented
        if(temp != target[curr_point]):
            num_incorrect += 1


        # current iteration information
        print ("Current iteration: " + str(total_runs))
        print ("Current Error: " + str(curr_err) + "\n")
        data.w(curr_err)

        # incrementing total runs and number of current point 
        total_runs += 1
        curr_point += 1

    data.close()
    print ("testing on given data is over check given two file for getting testing and training error info ('testing_err.txt' and 'training_err.txt')")
    print ("Percentage error on testing dataset: " + str(float(num_incorrect)/len(f)))
