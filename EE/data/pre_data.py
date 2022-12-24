import json

t_dict = dict()
N = 45

with open("duee_train.json", "w", encoding="utf-8") as w:
    with open("train.json", "r", encoding="utf-8") as t:
        lines = t.readlines()
        for line in lines:
            s = line
            line = json.loads(line.strip())
            events = line['event_list']
            for e in events:
                e_type = e['event_type']
                if t_dict.get(e_type, None) is None:
                    t_dict[e_type] = 0
                if t_dict[e_type] < N:
                    t_dict[e_type] += 1
                    w.write(s)
                break
print(t_dict)