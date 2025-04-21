def get_pred_step(uc):
    chat_comp = qianfan.ChatCompletion()

    if "uctext" in uc:
        resp = chat_comp.do(model="ERNIE-4.0-Turbo-8K", messages=[{
            "role": "user",
            "content":
                "You are an expert in software requirement analysis and software design. "
                "Please design the functional steps to implement the use case based on the given use case name and use case content description:"
                "Use case name:" + uc["ucName"] + "Use case content description:" + uc["uctext"] +
                ". Please put all functional steps into one string and output it, do not output any other information, no serial numbers, no line breaks, and do not output Chinese."
        }], disable_search=True)

    else:
        resp = chat_comp.do(model="ERNIE-4.0-Turbo-8K", messages=[{
            "role": "user",
            "content":
                "You are an expert in software requirement analysis and software design. "
                "Please design the functional steps to implement the use case based on the given use case name:"
                "Use case name:" + uc["ucName"] +
                ". Please put all functional steps into one string and output it, do not output any other information, no serial numbers, no line breaks, and do not output Chinese."
        }], disable_search=True)

    pred_steps = resp["body"]["result"]

    return pred_steps

def splite_2_list(str1):
    str1 = str1.translate(str.maketrans(REPLACEMENT_MAP))  # 先把中文符号都替换成英文的
    str2 = re.sub(r'\band\b|\bor\b', ',', str1)  # and和or 替换成逗号
    str1 = str2.split(",")
    clean_list = []
    for item in str1:
        if item != "" and not is_special_symbols(item):
            item = item.strip()  # 去除两端空格
            item = re.sub(r'^[\W_]+|[\W_]+$', '', item)  # 去除两端特殊符号
            clean_list.append(item)
    return clean_list

def is_exist(pred_node, ground_truth_list):
    chat_comp = qianfan.ChatCompletion()

    # 指定特定模型
    resp = chat_comp.do(model="ERNIE-4.0-Turbo-8K", messages=[{
        "role": "user",
        "content": "You are an expert in determining similarity. You are given two inputs: a single word A and a list of words B. "
                   "Your task is to determine whether any of the words in list B are either identical or similar to word A."
                   "Identical words include different forms of the same word (e.g., 'run' and 'running', 'dog' and 'dogs')."
                   "Similar words include those that are semantically similar, referring to the same action, entity, or concept, or are abbreviations or alternate representations (e.g., 'car' and 'automobile', 'NYC' and 'New York City')."
                   "Input: Word A:" + pred_node + ". List B:" + ground_truth_list +
                   ". Please output a boolean value: True if any word in list B is identical or similar to word A, otherwise False."
                   "Do not output any other information. Do not output Chinese."
    }], disable_search=True)

    exist = resp["body"]["result"]

    return str_to_bool(exist)

def pred_steps_and_write(use_case_list, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for uc in use_case_list:
            print(f'*** uc index: {uc["index"]}, dataset:{uc["dataset"]} ***')
            uc["pred_steps"] = get_pred_step(uc)

            for key, value in uc.items():
                if value is None:
                    print(f"错误！key '{key}' 的 value 为 None")

            f.write(json.dumps(uc, ensure_ascii=False) + '\n')

    return use_case_list

if __name__ == '__main__':
    # 0. 确认要完成的 task_index
    task_index = 4  # 选择需要完成的task
    print(f"******** start # {task_index} task !!********")

    # task 1: 使用 ernie 4 turbo 进行 pred step（需要与ernie speed 不同的调用函数)
    if task_index == 1:
        # 1. read uc from .json file
        use_case_list = read_uc_from_json(dataset_origin_node_path + pub_dataset_name)

        # 2. get prediction steps using Ernie 4 Turbo 8K
        use_case_list = pred_steps_and_write(use_case_list, dataset_pred_step_path + pub_pred_step)

    # task 2: 使用 ernie 4 turbo 对 pred step 进行 node 提取
    elif task_index == 2:
        # 1. read uc from .json file
        use_case_list = read_uc_from_json(dataset_pred_step_path + pub_pred_step)

        # 2. extract node
        use_case_list = extract_pred_node(use_case_list, dataset_pred_node_path + pub_pred_node)


    # task 3: 根据uc中的ground truth和pred node, 找出tp(true positive)(这一版是没有分step存储node的tp提取，新版按step分的在实验室电脑
    # _controlled_exper_PE_based_public_dataset.py中task11）
    elif task_index == 3:
        # 1. read uc from .json file
        use_case_list = read_uc_from_json(dataset_pred_node_path + pub_pred_node)  # 上个版没有按照step分别存放node

        # 2. 因为现在的node都是str，需要变成list，并且删除其中空格等
        with open(dataset_pred_node_path + "with_tp/" + pub_pred_node, 'a', encoding='utf-8') as f:
            for uc in use_case_list:
                print(f"uc index: {uc['index']}, uc dataset: {uc['dataset']}")
                uc["act"] = splite_2_list(uc["act"])
                uc["obj"] = splite_2_list(uc["obj"])
                uc["pred_act"] = splite_2_list(uc["pred_act"])
                uc["pred_obj"] = splite_2_list(uc["pred_obj"])

                uc["tp"] = []
                for pred_act in uc["pred_act"]:
                    if is_exist(pred_act, str(uc["act"])):
                        uc["tp"].append(pred_act)
                for pred_obj in uc["pred_obj"]:
                    if is_exist(pred_obj, str(uc["obj"])):
                        uc["tp"].append(pred_obj)

                f.write(json.dumps(uc, ensure_ascii=False) + '\n')

    print(f'****** Finish !!! ******')
