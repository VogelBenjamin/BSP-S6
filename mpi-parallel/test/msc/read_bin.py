import struct

size = 420
values = 4140*2-420

def read_binary_file(filename):
    with open(filename, "rb") as file:
        raw_bytes = file.read(3*4)
        integer_values = struct.unpack("3i", raw_bytes)
        print(integer_values)

        raw_bytes = file.read(420*4)
        integer_values = struct.unpack("420i", raw_bytes)

        raw_bytes = file.read(7860*4)
        integer_values = struct.unpack("7860i", raw_bytes)

        raw_bytes = file.read(7860*8)
        double_values = struct.unpack("7860d", raw_bytes)
        print(double_values[0])

        


        
    
    
# Example usage
filename = "binary_mat.bin"  # Replace with the actual filename
read_binary_file(filename)