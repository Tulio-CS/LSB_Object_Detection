import csv

header = ['y'] + [f'Head_{i}_{axis}' for i in range(468) for axis in ['x', 'y', 'z']] + [f'Pose_{i}_{axis}' for i in range(33) for axis in ['x', 'y', 'z',"visibility"]] + [f'Hand_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]

with open("teste.csv","w") as file:
    writer = csv.writer(file)
    writer.writerow(header)

    file.close()