
from helper import MemoryProfile

if __name__ == '__main__':
     
    # set paths to prof files from jax
    prof_file_0 = 'prof/memory_pre_solve.prof'
    prof_file_1 = 'prof/memory_post_solve.prof'
    
    # parse the prof files
    pprof = MemoryProfile(prof_file_1, base=prof_file_0)

    # get properties
    cpu_memory = pprof.get_device_memory('cpu')
    gpu_memory = pprof.get_device_memory('cuda:0')
    
    # print results
    print(f'CPU Memory: {cpu_memory}')
    print(f'GPU Memory: {gpu_memory}')
