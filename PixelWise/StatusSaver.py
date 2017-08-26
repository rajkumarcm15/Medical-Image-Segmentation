import numpy as np

class StatusSaver:

    def __init__(self, file_name, objects):
        # file = open(file_name,'a')
        for obj in objects:
            # if type(obj) is np.ndarray:
            #     self.print_array(file, obj)
            # else:
            #     self.print_str(file, obj)
            print(obj)
        # file.close()
    
    def print_array(self, file, obj):
        np.savetxt(file, obj, fmt='%.2f')
        file.write('\n')

    def print_str(self, file, obj):
        file.write(obj)
        file.write('\n')
