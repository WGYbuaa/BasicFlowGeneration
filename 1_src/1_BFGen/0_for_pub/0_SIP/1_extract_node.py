# 读取用dict保存的uc
def read_uc_from_json(file_path):
    use_case_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # line = line.replace("'", '"')  # 有时候数据集中会有多余的单引号或者双引号
            try:
                uc = json.loads(line)
                use_case_list.append(uc)
            except json.JSONDecodeError as e:
                print(f"错误信息: {e},  Line: {inspect.currentframe().f_lineno}, json读取失败")
    return use_case_list

# 后面换简单的prompt了，详见DELL G7 utils_gpt.py 的 extract_act_eng_simplified()
def extract_act_eng(steps):
    chat_comp = qianfan.ChatCompletion()
    # 指定特定模型
    speed = "ERNIE-Speed-8K"
    turbo = "ERNIE-4.0-Turbo-8K"
    ernie35 = "ERNIE-3.5-8K-0701"

    resp = chat_comp.do(model=turbo, messages=[{
        "role": "user",
        "content": "You are an expert in software requirement analysis and action extraction."
                   "Please extract all the action of the sentence from the given requirement specification sentence: "
                   + steps +
                   ". If there is no action, output the predicate of the sentence. If there are duplicate actions, output them in order."
                   "Please join all verbs or predicates with commas and output them as strings, do not output any other information, no serial numbers, no line breaks, and do not output Chinese."
    }], disable_search=True)

    act_list = resp["body"]["result"]

    act_list = re.sub(r'\n', ', ', act_list)
    act_list = re.sub(r'(\d+)\.', ', ', act_list)

    return act_list

def extract_obj_eng(steps):
    chat_comp = qianfan.ChatCompletion()
    # 指定特定模型
    speed = "ERNIE-Speed-8K"
    turbo = "ERNIE-4.0-Turbo-8K"
    ernie35 = "ERNIE-3.5-8K-0701"

    resp = chat_comp.do(model=turbo, messages=[{
        "role": "user",
        "content": "You are an expert in software requirement analysis and entity extraction."
                   "Please extract all the entity of the sentence from the given requirement specification sentence: "
                   + steps +
                   ". If there is no entity, output the object of the sentence. If there are duplicate entities, output them in order."
                   "Please join all entities or objects with commas and output them as strings, do not output any other information, no serial numbers, no line breaks,and do not output Chinese."
    }], disable_search=True)

    obj_list = resp["body"]["result"]

    obj_list = re.sub(r'\n', ', ', obj_list)
    obj_list = re.sub(r'(\d+)\.', ', ', obj_list)

    return obj_list

def extract_node_step_by_step(use_case_list, out_path):
    with open(out_path, 'a', encoding='utf-8') as f:
        for uc in use_case_list:
            print(f'*** uc index: {uc["index"]}, dataset:{uc["dataset"]} ***')
            uc["act"], uc["obj"] = [], []  # 清空之前提取的
            # act和obj分开放。不用断句，要求提取所有的act和obj。
            if isinstance(uc["steps"], str):
                print(f' uc steps 格式出错！！')  # 都已经转换成了列表
                uc["act"] = extract_act_eng(uc["steps"])
                uc["obj"] = extract_obj_eng(uc["steps"])
            elif isinstance(uc["steps"], list):
                for s in uc["steps"]:
                    uc["act"].append(extract_act_eng(s))
                    uc["obj"].append(extract_obj_eng(s))
            else:
                print(f' uc steps 格式出错！！')

            for key, value in uc.items():
                if value is None:
                    print(f"错误！key '{key}' 的 value 为 None, Line: {inspect.currentframe().f_lineno}")
            if len(uc['steps']) != len(uc['act']) or len(uc['steps']) != len(uc['obj']):
                print(f"uc index: {uc['index']} 提取次数不对！！！")

            f.write(json.dumps(uc, ensure_ascii=False) + '\n')

    return use_case_list




if __name__ == '__main__':

    # 确认要完成的 task_index
    print("task0-6中很多是操作ernie speed 8k 模型时用的，task7开始改成了 ernie 4 Turbo 8k/ 3.5 ！！ ")
    task_index = 16  # 选择需要完成的task
    print(f"******** start # {task_index} task !!********", datetime.now().strftime("%H:%M:%S"))
    if task_index == 10:
            seg_path = "ControlledExper/4_dataset_pred_node/ERNIE_4_Turbo_8k/with_tp/2nd_round/step_seg/Ernie_pub_seg.json"
            out_path = "ControlledExper/4_dataset_pred_node/ERNIE_4_Turbo_8k/with_tp/3rd_round/Ernie_pub.json"
            use_case_list = read_uc_from_json(seg_path)

            # 1. 逐句提取node，并且直接写入
            use_case_list = extract_node_step_by_step(use_case_list, out_path)