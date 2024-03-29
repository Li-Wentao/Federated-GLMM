from GLMM_LA import *
##################### Simulation with Penn ############################
import os
truth = np.array([-1.5,0.1,-0.5,-0.3,0.4,-0.2,-0.25,0.35,-0.1,0.5]).reshape(10, 1)
var_name = []
for i in range(10):
    var_name += ['X' + str(i+1)]
import sys

# Setting 6
print('======================\n Here starts Setting 6 \n======================\n\n\n')

file_dir = '../../Simulation_data_GLMM/Setting_6/'
file_names = os.listdir(file_dir)
for i in range(len(file_names)):
    try:
        start_time = time.time()
        df = pd.read_csv(file_dir + file_names[i], index_col=0)
        # Load data
        data1 = np.array(df[df['Site_ID'] == 1][var_name])
        data2 = np.array(df[df['Site_ID'] == 2][var_name])
        data = [data1, data2]
        out1 = np.array(df[df['Site_ID'] == 1]['y']).reshape(30,1)
        out2 = np.array(df[df['Site_ID'] == 2]['y']).reshape(30,1)
        out = [out1, out2]
        # Simulations
        [beta, mu] = LA(data, out)
        t = time.time() - start_time
        output(data, beta, truth, t).to_csv('../../Simulation_data_GLMM/Result_LA/Setting_6_' +
                                              file_names[i][43:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;