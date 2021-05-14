from GLMM_GH import *
##################### Simulation with Penn ############################
import os
truth = np.array([-1.5,0.1,-0.5,-0.3,0.4,-0.2,-0.25,0.35,-0.1,0.5]).reshape(10, 1)
var_name = []
for i in range(10):
    var_name += ['X' + str(i+1)]
k = 3

# Setting 8
print('======================\n Here starts Setting 8 \n======================\n\n\n')

file_dir = '../../Simulation_data_GLMM/Setting_8/'
file_names = os.listdir(file_dir)
for i in range(len(file_names)):
    try:
        start_time = time.time()
        df = pd.read_csv(file_dir + file_names[i], index_col=0)
#         Load data
        data = []
        for k in range(10):
            globals()['data%s' % str(k+1)] = np.array(df[df['Site_ID'] == k+1][var_name])
            data.append(globals()['data%s' % str(k+1)])
        out = []
        for k in range(10):
            globals()['out%s' % str(k+1)] = np.array(df[df['Site_ID'] == k+1]['y']).reshape(30,1)
            out.append(globals()['out%s' % str(k+1)])
        # Simulations
        [beta, mu] = GH(k, data, out)
        output(data, beta, truth).to_csv('../../Simulation_data_GLMM/Result_GH/Setting_8_' +
                                              file_names[i][44:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;