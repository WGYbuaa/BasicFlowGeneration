# Use ERNIE to get basic flow and perform node extraction.

def read_uc_from_json_NCET(file_path):
    use_case_list = list()  # 按顺序存放所有的用例uc
    use_case = useCase()
    uc_number = 0  # uc的全局index
    step_number = 0  # step在uc内的位置

    with open(file_path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()  # 移除可能的空格
            if len(line) == 0:  # 数据集的结尾需要多一个空格，才会返回最后一个uc
                if len(use_case.step_former) > 0:
                    use_case.index_global = uc_number
                    use_case.tc_text = tc_text
                    use_case.tc_path = tc_path
                    use_case_list.append(use_case)  # 测试用例的总和，按照一个个TC存放
                    use_case = useCase()
                    uc_number = uc_number + 1  # 计算为下一个use case的index做准备
                    step_number = 0

                    # if len(use_case_list) == 2:  # 先用前三十个use case 做demo。不用的时候注释掉这两行即可
                    #     return use_case_list

                continue

            json_data = json.loads(line)  # 以行为单位,存入json_data中

            if 'index_global' not in json_data:
                (tc_text, tc_path) = json_data
            else:
                # 导入新的use_case_step中，包括以下内容：
                use_case_step = useCaseStep(json_data['index_global'], json_data['step'],
                                            json_data['parameter'], json_data['returns'], uc_number, step_number)
                step_number += 1  # 下一个step的index

                use_case.add_ts(use_case_step, use_case_step.step)

    return use_case_list


def extract_act_cn(steps):
    chat_comp = qianfan.ChatCompletion()

    # 指定特定模型
    resp = chat_comp.do(model="ERNIE-4.0-Turbo-8K", messages=[{
        "role": "user",
        "content": "你是软件需求分析和动作提取方面的专家，请从下面给定的需求规格语句中提取所有的动词："
                   + steps +
                   "。如果有重复的动词，则按顺序输出这些动词，不要去重；如果没有动词，则输出句子的谓语；如果都没有，则输出空字符串。"
                   "请将所有动词或谓语用逗号间隔，放入一个字符串并输出，不要输出任何其他信息和描述性语句，不要输出序号，不要换行，不要输出除中文外的其他语言。"
    }], disable_search=True)

    act_list = resp["body"]["result"]

    act_list = re.sub(r'\n', ', ', act_list)
    act_list = re.sub(r'(\d+)\.', ', ', act_list)

    return act_list

def extract_obj_cn(steps):
    chat_comp = qianfan.ChatCompletion()

    # 指定特定模型
    resp = chat_comp.do(model="ERNIE-4.0-Turbo-8K", messages=[{
        "role": "user",
        "content": "你是软件需求分析和实体提取方面的专家，请从下面给定的需求规格语句中提取所有的实体："
                   + steps +
                   "。如果有重复的实体，则按顺序输出这些实体，不要去重；如果没有实体，则输出句子的宾语；如果都没有，则输出空字符串。"
                   "请将所有实体或宾语用逗号间隔，放入一个字符串并输出，不要输出任何其他信息和描述性语句，不要输出序号，不要换行，不要输出除中文外的其他语言。"
    }], disable_search=True)

    obj_list = resp["body"]["result"]

    obj_list = re.sub(r'\n', ', ', obj_list)
    obj_list = re.sub(r'(\d+)\.', ', ', obj_list)

    return obj_list
    
def extract_node_from_pred(use_case_list, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for uc in use_case_list:
            print(f" uc index: {uc['index']} .")
            uc["pred_act"], uc["pred_obj"] = "",""

            act = extract_act_cn(uc["pred_steps"])
            uc["pred_act"] = act
            obj = extract_obj_cn(uc["pred_steps"])
            uc["pred_obj"] = obj


            for key, value in uc.items():
                if value is None:
                    print(f"错误！key '{key}' 的 value 为 None, Line: {inspect.currentframe().f_lineno}")

            f.write(json.dumps(uc, ensure_ascii=False) + '\n')
def get_pred_step_cn(uc):
    chat_comp = qianfan.ChatCompletion()

    resp = chat_comp.do(model="ERNIE-4.0-Turbo-8K", messages=[{
        "role": "user",
        "content":
            "你是软件需求分析和软件设计方面的专家，请根据给定的用例描述和用例存放的路径，设计功能步骤："
            "用例描述：" + uc.tc_text + "。用例存放的路径：" + uc.tc_path +
            "。请将所有功能步骤用句号连接，放入一个字符串并输出，不要输出任何其他信息和描述性语句，不要输出序列号，不要换行，不要输出除中文外的其他语言。"
    }], disable_search=True)

    pred_steps = resp["body"]["result"]

    return pred_steps


def pred_steps_and_write_cn(use_case_list, out_path):
    update_uc = {}
    uc_list = []
    with open(out_path, 'w', encoding='utf-8') as f:
        for uc in use_case_list:
            print(f" uc index: {uc.index_global} .")
            update_uc["index"] = uc.index_global
            update_uc["pred_steps"] = []
            update_uc["dataset"] = "NCE-T"
            update_uc["ucText"] = uc.tc_text
            update_uc["ucPath"] = uc.tc_path

            update_uc["pred_steps"] = get_pred_step_cn(uc)

            for key, value in update_uc.items():
                if value is None:
                    print(f"错误！key '{key}' 的 value 为 None")

            uc_list.append(update_uc)

            f.write(json.dumps(update_uc, ensure_ascii=False) + '\n')

    return uc_list

def read_uc_from_json(file_path):
    use_case_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                uc = json.loads(line)
                use_case_list.append(uc)
            except json.JSONDecodeError as e:
                print(f"错误信息: {e},  Line: {inspect.currentframe().f_lineno}")
    return use_case_list


if __name__ == '__main__':
    # 0. 确认要完成的 task_index
    task_index = 1  # 选择需要完成的task
    print(f"******** start # {task_index} task !!********")

    # task 0: 使用 ernie 4 turbo 进行 pred step（需要与ernie speed 不同的调用函数)
    if task_index == 0:
        print("************* Step 2 ***********", datetime.now().strftime("%H:%M:%S"))
        file_path = "../data/1_dataset_origin/2_uc_json_origin/NCE-T_tc.json"
        out_path = "../data/3_dataset_prediction_step/Ernie_4_Turbo/Ernie_NCE-T_steps.json"


        # 1. 读取中文uc，read_uc_from_json_NCET()函数节选自extract_use_case_step_1()， 删除了act/obj/step_clean部分。
        use_case_list = read_uc_from_json_NCET(file_path)

        # 2. get prediction steps using Ernie 4 Turbo 8K
        use_case_list = pred_steps_and_write_cn(use_case_list, out_path)

    # task 1: 使用 ernie 4 turbo 从 pred step 中提取node
    if task_index == 1:
        print("************* Step 2 ***********", datetime.now().strftime("%H:%M:%S"))
        file_path = "../data/3_dataset_prediction_step/Ernie_4_Turbo/Ernie_NCE-T_steps.json"
        out_path = "../data/4_dataset_pred_node/Ernie_NCE-T.json"


        # 1. 读取uc
        use_case_list = read_uc_from_json(file_path)


        # 2. 从pred step中提取node
        use_case_list = extract_node_from_pred(use_case_list, out_path)

    print(f'****** Finish !!! ******')
