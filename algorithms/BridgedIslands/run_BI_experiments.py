import subprocess
from timeit import default_timer as timer
import os
import shutil
import json

# This script executes the original BI code for a number of datasets. It has been designed to do cross-validation,
# because it wasnt specifically designed to do so in the original.
#
# This script has been specifically designed for the article. Changes in the datasets or in the number of folds could be
# made by simply changing a couple of lines.

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
        command = ['java', '-Xmx1024M', '-cp', 'BI.jar', 'clustering/LearnAndTest',
                   '../../data/discrete/'+data+'/'+str(numberOfFolds)+'_folds/'+data+'_'+str(i)+'_train.arff',
                   '../../data/discrete/'+data+'/'+str(numberOfFolds)+'_folds/'+data+'_'+str(i)+'_test.arff',
                   './temp',
                   "32", "32", "32", "32", "64", "100", "0.01", "3"]

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
            os.rename("temp/FinalBIModel.bif", 'temp/'+data+'_'+str(i)+'.bif')
        else:
            os.rename("temp/FinalBIModel.bif", 'temp/'+data+'_'+str(i)+'.bif')

        # Results have been posted by BI.jar in the 'Result_LearnAndTest.txt' file,
        # we now load them and post them in the results.json file
        ll = 0
        bic = 0
        with open("temp/Result_LearnAndTest.txt", "r") as file:
            for line in file:
                if line.startswith("Loglikelihood (base e) on the testing data:"):
                    ll = float(line.replace("Loglikelihood (base e) on the testing data:", ""))
                elif line.startswith("BIC score on the testing data:"):
                    bic = float(line.replace("BIC score on the testing data:", ""))

        fold_results = {}
        print("Learning time = " + str(learning_time))
        fold_results["time"] = learning_time
        print("Test LL = " + str(ll))
        fold_results["LL"] = ll
        print("Test BIC = " + str(bic))
        fold_results["BIC"] = bic

        #  Move the result model
        if os.path.isfile('results/' + data + '/' + data + '_' + str(i) + '.bif'):
            os.remove('results/' + data + '/' + data + '_' + str(i) + '.bif')
            shutil.move('temp/' + data + '_' + str(i) + '.bif', 'results/' + data + '/' + data + '_' + str(i) + '.bif')
        else:
            shutil.move('temp/' + data + '_' + str(i) + '.bif', 'results/' + data + '/' + data + '_' + str(i) + '.bif')

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