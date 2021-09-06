from GLMM_GH import *
##################### Simulation with Covid Data ############################
import os
import sys

file_dir = '../../Covid_data_GLMM/Distributed/'
file_names = os.listdir(file_dir)
start_time = time.time()
data = []
out = []
k = 2
for f in file_names:
	df = pd.read_csv(file_dir + f, index_col=0)
	# Add an intercept
	df.insert(loc=0, column='Intercept', value=1)
	data_ans = np.array(df.drop('target', axis=1))
	out_ans = np.array(df.target).reshape(len(df),1)
	data.append(data_ans)
	out.append(out_ans)
var_name = df.drop('target', axis=1).columns

# Regression
[beta, mu, lam, score] = GH(k, data, out)
t = time.time() - start_time
output(data, beta, t, var_name).to_csv('../../Covid_data_GLMM/Result_GH/result_gh.csv')
aic = -2 * score + 2 * len(beta)
bic = -2 * score + np.log(46312) * len(beta)
with open('../../Covid_data_GLMM/Result_GH/gh_stats.txt', 'w') as f:
    f.write('Mu: \n' + str(mu))
    f.write('\n\nLambda: \n' + str(lam))
    f.write('\n\nLog-likelihood: \n' + str(score))
    f.write('\n\nAIC: \n' + str(aic))
    f.write('\n\nAIC: \n' + str(bic))
    f.write('\n\nRun time: \n' + str(time.time() - start_time) + ' (seconds)')
# pd.DataFrame(mu).to_csv('../../Covid_data_GLMM/Result_GH/mu.csv')
# print('The lambda is: \n', lam)
# print('The Log-Likelihood is: \n', lam)
print("--- %s seconds ---" % (time.time() - start_time))






