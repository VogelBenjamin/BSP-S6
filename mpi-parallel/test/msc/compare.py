def read_floats_from_file(filename):
    with open(filename, 'r') as file:
        return list(map(float, file.read().split()))

def compare_files(file1, file2):
    data1 = read_floats_from_file(file1)
    data2 = read_floats_from_file(file2)

    differences = []

    for i in range(min(len(data1), len(data2))):
        if data1[i] != data2[i]:
            differences.append((i, data1[i], data2[i]))

    return differences

file1 = "r_mpi.txt"  # Replace with actual file name
file2 = "r_serial.txt"  # Replace with actual file name

differences = compare_files(file1, file2)

if differences:
    print("Differences found:")
    for index, value1, value2 in differences:
        print(f"Index {index}: {value1} != {value2}")
else:
    print("The sequences are identical.")