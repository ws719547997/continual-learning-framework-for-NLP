"""
9.24 把txt的列表转化成json容纳更多信息
"""
import json

root_path = 'stock.txt'
output_path = 'stock.json'

task_list = []
task_json = {}
with open(root_path, 'r', encoding='utf-8') as f:
    task_list = [l.strip() for l in f]

task_json.update({'target':'linear'})
task_json.update({'task_list':[]})
for task in task_list:
    task_json['task_list'].append({
        'task_name':task,
        'task_type':'dsc',
        'language':"zh",
        'task_output':2,
        'target':'linear'
    })

with open(output_path,'w',encoding='utf-8') as f:
    f.write(json.dumps(task_json,indent=4,ensure_ascii=False))
