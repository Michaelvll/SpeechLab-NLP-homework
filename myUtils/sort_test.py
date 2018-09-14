with open('test.txt', 'r') as infile:
    lines = infile.readlines()
origin_order = {}
for idx, line in enumerate(lines):
    origin_order[line.strip()] = idx

with open('solution.txt', 'r') as infile:
    solution_lines_origin = [line.strip() for line in infile.readlines()]


def get_line(line):
    new_line = []
    for item in line.split(' <=> ')[0].split(' '):
        new_line.append(item.split(':')[0])
    return ' '.join(new_line)


new_solution = sorted(solution_lines_origin,
                      key=lambda line: origin_order[get_line(line)])
with open('new_soltion.txt', 'w', newline='\n') as outfile:
    for line in new_solution:
        print(line, file=outfile)
