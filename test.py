import csv
from datetime import datetime

time_step = []
tickets = []

with open("claims.txt", "r", encoding="utf-16") as f:
    csv_reader = csv.reader(f, delimiter=",")

    # Skip the header
    next(csv_reader)

    # Skip lines with NUL characters
    lines = (line for line in csv_reader if "\0" not in line)

    # Iterate through non-null lines
    for line in lines:
        # Assuming the first column is the date and the second column is the number
        time_step.append(datetime.strptime(line[0], "%Y-%m-%d"))
        tickets.append(float(line[1]))

# View the first 10 of each
print(time_step[:10], tickets[:10])
print(len(time_step))
