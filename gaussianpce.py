import multiprocessing
import scipy.io as sio
import chaospy as cp



skull=cp.Normal(2813.7,337)
nodes,weights = cp.generate_quadrature(2, skull, rule="gaussian" )
print(nodes)
mat_contents = sio.loadmat("C:\\Users\\Yunus\\Desktop\\pyy\\patSkullthree.mat")
evals = mat_contents['data']
#evals should be the output with respect to input nodes.
expansions = cp.generate_expansion(2, skull,cross_truncation=1)

def process_iteration(i):
    # seperate simulaton results for each voxel
    evalx = evals[:, (i)].flatten().tolist()
    gauss_model_approx = cp.fit_quadrature(expansions, nodes, weights, evalx,retall=1)
    print(i)
    return gauss_model_approx[1]#return coeff of respected voxel


def main():
    # Set the start method to 'spawn'
    multiprocessing.set_start_method('spawn')


if __name__ == "__main__":
    main()
    num_processes = 20# Set the number of processes based on your available CPUs
    indices = range(3891652)
    # Create a multiprocessing pool with the specified number of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use the 'map' method to apply the 'process_iteration' function to each index
        coeffs = pool.map(process_iteration, indices)
        mat_dict = {'data': coeffs}

        # specify the path where you want to save the file
        path = 'C:/Users/Yunus/Desktop/pyy/'
        # save the dictionary to a .mat file in the specified path
        sio.savemat(path + 'onlyskullgaussxianfuncwiththree' + '.mat', mat_dict)
        print("Data saved2.")


