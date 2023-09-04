import multiprocessing
import scipy.io as sio
from sklearn.linear_model import LinearRegression
import chaospy as cp

#create polynomials
polynomials = cp.monomial(
    start=0,
    stop=3,
    dimensions=20,
    cross_truncation=1,
    graded=True,
    reverse=True,
)
print(polynomials.size)
#simulation results
mat_contents = sio.loadmat("C:\\Users\\Yunus\\Desktop\\pyy\\model4results.mat")
evals = mat_contents['data']
mat_contents = sio.loadmat("C:\\Users\\Yunus\\Desktop\\pyy\\model4samples.mat")
samples = mat_contents['data']



numberofsampleused =20
#combine input of simulations
abscissas = [samples[0][:numberofsampleused],samples[1][:numberofsampleused],samples[2][:numberofsampleused],samples[3][:numberofsampleused],samples[4][:numberofsampleused],samples[5][:numberofsampleused],samples[6][:numberofsampleused],samples[7][:numberofsampleused]]


def process_iteration(i):
    #seperate simulaton results for each voxel
    evalx = evals[:numberofsampleused, (i)].flatten().tolist()
    model = LinearRegression(fit_intercept=False)
    function = cp.fit_regression(polynomials, abscissas, evalx, model=model, retall=1)
    print(i) # to see when process will end
    return function[1] #return coeff of respected voxel


def main():
    # Set the start method to 'spawn'
    multiprocessing.set_start_method('spawn')

if __name__ == "__main__":
    main()

    num_processes = 20  # Set the number of processes based on your available CPUs
    indices = range(31232)
    # Create a multiprocessing pool with the specified number of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use the 'map' method to apply the 'process_iteration' function to each index
        coeffs = pool.map(process_iteration, indices)
        mat_dict = {'data': coeffs}

        # specify the path where you want to save the file
        path = 'C:/Users/Yunus/Desktop/pyy/'

        # save the dictionary to a .mat file in the specified path

        sio.savemat(path + 'model4funcsdssdwith' + str(numberofsampleused) + '.mat', mat_dict)
        print("Data saved.")


