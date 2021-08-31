from tensorflow.python.client import device_lib

"""

Simple function to test if CPU / GPU devices can be detected by tensorflow.
Often the GPU will not be detected. CUDA / tensorflow-GPU is required for this.
Read more about that here: https://www.tensorflow.org/install/gpu

"""

def hasGPU():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
    
def hasCPU():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']

if __name__ == "__main__":
    print(list(device_lib.list_local_devices()))
    print(hasGPU())
