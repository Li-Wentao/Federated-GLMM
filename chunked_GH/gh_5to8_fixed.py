from GLMM_GH import *
##################### Simulation with Penn ############################
import os
truth = np.array([-1.5,0.1,-0.5,-0.3,0.4,-0.2,-0.25,0.35,-0.1,0.5]).reshape(10, 1)
var_name = []
for i in range(10):
    var_name += ['X' + str(i+1)]
k = 2

# Setting 5
print('======================\n Here starts Setting 5 \n======================\n\n\n')

file_dir = '../../Simulation_data_GLMM/Setting_5/'
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
        model = GH(k, data, out).fit()
        t = time.time() - start_time
        output_df = model.df
        output_df.insert(loc=0, column='Truth', value=truth)
        output_df['RunTime'] = t
        output_df['Steps'] = model.step
        output_df.to_csv('../../Simulation_data_GLMM/Result_GH_fixed/Setting_5_' +
                                              file_names[i][43:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;

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
        model = GH(k, data, out).fit()
        t = time.time() - start_time
        output_df = model.df
        output_df.insert(loc=0, column='Truth', value=truth)
        output_df['RunTime'] = t
        output_df['Steps'] = model.step
        output_df.to_csv('../../Simulation_data_GLMM/Result_GH_fixed/Setting_6_' +
                                              file_names[i][43:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;

# Setting 7
print('======================\n Here starts Setting 7 \n======================\n\n\n')

file_dir = '../../Simulation_data_GLMM/Setting_7/'
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
        model = GH(k, data, out).fit()
        t = time.time() - start_time
        output_df = model.df
        output_df.insert(loc=0, column='Truth', value=truth)
        output_df['RunTime'] = t
        output_df['Steps'] = model.step
        output_df.to_csv('../../Simulation_data_GLMM/Result_GH_fixed/Setting_7_' +
                                              file_names[i][44:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;

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
        model = GH(k, data, out).fit()
        t = time.time() - start_time
        output_df = model.df
        output_df.insert(loc=0, column='Truth', value=truth)
        output_df['RunTime'] = t
        output_df['Steps'] = model.step
        output_df.to_csv('../../Simulation_data_GLMM/Result_GH_fixed/Setting_8_' +
                                              file_names[i][44:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;


