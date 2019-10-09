import subprocess
from timeit import default_timer as timer
import os
import shutil
import json

# This script executes the original EAST code for a number of datasets. It has been designed to do cross-validation,
# because it wasnt specifically designed to do so in the original.
#
# This script has been specifically designed for the article. Changes in the datasets or in the number of folds could be
# made by simply changing a couple of lines.

#datasets = ["breast_w"]
datasets = ["asia", "breast_cancer", "breast_w", "hiv_test", "political_survey", "zoo"]
if not os.path.exists("temp"):
    os.makedirs("temp")
numberOfFolds = 10

results = {}
for data in datasets:
    print("\n" + data)
    print("--------------------------------------------")
    if not os.path.exists("results/"+data):
        os.makedirs("results/"+data)
    data_results = {}
    avg_time = 0
    avg_ll = 0
    avg_bic = 0
    data_results["average_learning_time"] = avg_time
    data_results["average_predictive_ll"] = avg_ll
    data_results["average_predictive_bic"] = avg_bic
    for i in range(1, 11):
        command = ['java', '-Xmx1024M', '-cp', 'east.jar', 'EAST', '4', '10','0.1', '50', '16', '20', '0.1', '64',
                   '100', '0.1', '../../data_east/'+data+'/'+str(numberOfFolds)+'_folds/'+data+'_'+str(i)+'_train.data', './temp']

        print("\nFold " + str(i) + ":")

        # Learn the model
        start = timer()
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        result = p.communicate()
        p.wait()
        end = timer()
        learning_time = (end - start) * 1000

        # Rename the learned file to BIF format for evaluation
        if os.path.isfile('temp/'+data+'_'+str(i)+'.bif'):
            os.remove('temp/'+data+'_'+str(i)+'.bif')
            os.rename("temp/M.BIC.txt", 'temp/'+data+'_'+str(i)+'.bif')
        else:
            os.rename("temp/M.BIC.txt", 'temp/'+data+'_'+str(i)+'.bif')

        # Evaluate the model using test data
        p = subprocess.Popen(['java', '-cp', 'east.jar', 'evaluateModel', 'temp/'+data+'_'+str(i)+'.bif', '../../data_east/'+data+'/'+str(numberOfFolds)+'_folds/'+data+'_'+str(i)+'_test.data'],
                             stdout=subprocess.PIPE)
        stdout, stderr = p.communicate()
        p.wait()

        # Store the results
        #   Score & learning time
        stdout = stdout.decode("utf-8")
        stdout = stdout.replace("BIC:", "")
        stdout = stdout.replace("LL:", "")
        stdout = stdout.split("\r\n")
        ll = float(stdout[len(stdout) - 3])
        bic = float(stdout[len(stdout) - 2])
        fold_results = {}
        print("Learning time = " + str(learning_time))
        fold_results["time"] = learning_time
        print("Test LL = " + str(ll))
        fold_results["LL"] = ll
        print("Test BIC = " + str(bic))
        fold_results["BIC"] = bic
        #   Resulting model
        if os.path.isfile('results/'+data+'/'+data+'_'+str(i)+'.bif'):
            os.remove('results/'+data+'/'+data+'_'+str(i)+'.bif')
            shutil.move('temp/'+data+'_'+str(i)+'.bif', 'results/'+data+'/'+data+'_'+str(i)+'.bif')
        else:
            shutil.move('temp/'+data+'_'+str(i)+'.bif', 'results/'+data+'/'+data+'_'+str(i)+'.bif')

        data_results["fold_" + str(i)] = fold_results
        avg_time = avg_time + learning_time
        avg_ll = avg_ll + ll
        avg_bic = avg_bic + bic

    avg_bic = avg_bic / numberOfFolds
    avg_ll = avg_ll / numberOfFolds
    avg_time = avg_time / numberOfFolds
    data_results["average_learning_time"] = avg_time
    data_results["average_predictive_ll"] = avg_ll
    data_results["average_predictive_bic"] = avg_bic

    results[data] = data_results

if os.path.isfile('results.json'):
    os.remove('results.json')
    with open('results.json', 'w') as fp:
        json.dump(results, fp, sort_keys=True, indent=4)
else:
    with open('results.json', 'w') as fp:
        json.dump(results, fp, sort_keys=True, indent=4)

input("All the experiments have finished. Models resulting of each fold can be found in the 'results' folder. "
      "The score and time results can be found in 'results.json'")

