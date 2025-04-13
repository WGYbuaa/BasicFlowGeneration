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

# 调用ernie断句
def seg_sent_eng_ernie(step_str):
    chat_comp = qianfan.ChatCompletion()

    # 指定特定模型
    speed = "ERNIE-Speed-8K"
    turbo = "ERNIE-4.0-Turbo-8K"
    ernie35 = "ERNIE-3.5-8K-0701"  # 换用3.5试试看

    str1 = "str1='" + step_str + "'.Please break 'str1' into multiple sentences and output them as a list,with each item surrounded by double quotes. Do not output code and explanatory statements."

    # 输入llm
    resp = chat_comp.do(model=ernie35, messages=[{
        "role": "user",
        "content": str1
    }], disable_search=True)

    exist = resp["body"]["result"]
    exist = exist.translate(str.maketrans(REPLACEMENT_MAP))

    # 使用正则表达式匹配方括号及其中的内容
    match = re.search(r'\[(.*?)\]', exist)
    # 如果找到了匹配项
    if match:
        # 提取匹配项中的内容，并去掉首尾的空格
        content = match.group(0).strip()
        print(f"content: {content}")
        list1 = ast.literal_eval(content)
        if isinstance(list1, list):
            return list1
        else:
            return "false"
    elif "false" in exist:
        return "false"
    else:
        return None

# pub数据集中的steps需要断句
def steps_seg(use_case_list, out_path, key):
    max_count = 11  # 最大调用ernie次数

    with open(out_path, 'a', encoding='utf-8') as f:
        for uc in use_case_list:
            print(f'uc index: {uc["index"]}')
            if key not in uc.keys():
                f.write(json.dumps(uc, ensure_ascii=False) + '\n')
                continue
            if isinstance(uc[key], str):
                count = 0
                while True:  # 无限循环，直至输出false或者
                    steps = seg_sent_eng_ernie(uc[key])
                    if steps and (isinstance(steps, list) or "false" in steps):
                        break
                    count += 1
                    if count >= max_count:
                        print(" 循环已经到达最大次数，退出循环 ")
                        break

                if "false" not in steps:  # 成功分割steps
                    if isinstance(steps, list):
                        uc[key] = steps
                    else:
                        uc[key] = " 该uc的steps需要重新断句！！！ "

            if isinstance(uc[key], list):  # 检验格式，如果是list再写
                f.write(json.dumps(uc, ensure_ascii=False) + '\n')
            else:
                print(f'uc index: {uc[key]} 格式不是list')
    return use_case_list


if __name__ == '__main__':

    # 确认要完成的 task_index
    print("task0-6中很多是操作ernie speed 8k 模型时用的，task7开始改成了 ernie 4 Turbo 8k/ 3.5 ！！ ")
    task_index = 16  # 选择需要完成的task
    print(f"******** start # {task_index} task !!********", datetime.now().strftime("%H:%M:%S"))
    if task_index == 7:
            read_path = "ControlledExper/2_dataset_origin_node/Ernie-4-Turbo/with_keyword/Ernie_pub_ground_truth.json"
            out_path = "ControlledExper/4_dataset_pred_node/ERNIE_4_Turbo_8k/with_tp/2nd_round/step_seg/Ernie_pub_seg.json"
            # 1. 读取
            use_case_list = read_uc_from_json(read_path)

            # 2. 调用ernie 给step断句
            use_case_list = steps_seg(use_case_list, out_path)