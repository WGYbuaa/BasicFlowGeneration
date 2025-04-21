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

def format_key_node(use_case_list):
    for uc in use_case_list:
        uc['key_act'] = splite_2_list_cn(uc['key_act'])
        uc['key_obj'] = splite_2_list_cn(uc['key_obj'])
        uc['key_path'] = splite_2_list_cn(uc['key_path'])
    return use_case_list


def delete_error(list1, list2):
    # 1、删除空项、空格、或者仅包含符号的项
    symbols_to_check = set('!@#$%^&*()"')  # 定义一个符号集合，用于检查元素中是否包含这些符号

    # 使用列表推导式来过滤列表
    list1 = [item for item in list1 if not (
            not item  # 检查是否为空字符串
            or any(symbol in item for symbol in symbols_to_check)  # 检查是否包含任何指定符号
    )]

    # 2、删除error node
    filtered_list1 = []

    # 遍历list1中的每个元素
    for item in list1:
        # 检查item中是否包含list2中的任意字符串
        contains_substring = any(sub_str in item for sub_str in list2)
        # 如果不包含，则将其添加到过滤后的列表中
        if not contains_substring:
            filtered_list1.append(item)
    return filtered_list1

def check_argc(use_case_list_dict):
    # 1.1 统计所有的node数目（act\obj\key）
    if isinstance(use_case_list_dict[0]["act"][0], list):  # 如果是嵌套列表，即node已经按照step分好了。（chatgpt版本此处已经分好）
        argc = count_node_data_dict_all_nested(use_case_list_dict)
    elif isinstance(use_case_list_dict[0]["act"][0], int):  # 如果不是，则node没有按照step分好。（Ernie版本此处还不需要）
        argc = count_node_data_dict_all(use_case_list_dict)
    else:
        print(f'ERROR: check_argc!!')
    # # 1.2 为了统计edge数目，生成一次全局edge_list
    # edge_dict, node_to_UCText = create_edges_llm(use_case_list_dict, ERROR_WORD_LIST, use_case_list_ori)
    #
    # act, obj, key, edges_num, edges_deduplicate = count_node_sub_data(edge_dict)
    # if len(act) != argc['act_node']:
    #     print("act node 全局数目对不上！！")
    # if len(obj) != argc['obj_node']:
    #     print("obj node 全局数目对不上！！")
    # if len(key) != argc['key_node']:
    #     print("key node 全局数目对不上！！")
    # argc['edges_num'] = edges_num
    # argc['edges_deduplicate'] = edges_deduplicate
    return argc

def group_uc_fixed(use_case_list, group_uc):
    use_case_list_new = []
    for index_list in group_uc:
        sub_uc_list = []
        for index in index_list:
            sub_uc_list.append(use_case_list[index])
        use_case_list_new.append(sub_uc_list)
    return use_case_list_new

def delete_error_node_4turbo(uc, error_word_list):
    uc["act"] = delete_error(uc["act"], error_word_list)
    uc["obj"] = delete_error(uc["obj"], error_word_list)

    if "key_path" in uc.keys():
        uc["key_path"] = delete_error(uc["key_path"], error_word_list)
    if "key_name" in uc.keys():
        uc["key_name"] = delete_error(uc["key_name"], error_word_list)

    uc["key_act"] = delete_error(uc["key_act"], error_word_list)
    uc["key_obj"] = delete_error(uc["key_obj"], error_word_list)

    return uc

# 用于llm as enhancer（act/obj没有按照step存放）
def get_edges_dict_and_node_to_UCText_4turbo(use_case_list, ERROR_WORD_LIST, use_case_list_ori):
    edges_dict_list = []
    node_to_UCText_list = []
    for uc_list, uc_list_ori in zip(use_case_list, use_case_list_ori):
        edge_dict, node_to_UCText = create_edges_llm(uc_list, ERROR_WORD_LIST, uc_list_ori)
        edges_dict_list.append(edge_dict)
        node_to_UCText_list.append(node_to_UCText)

    return edges_dict_list, node_to_UCText_list


def get_keyword_pub_dataset(use_case_list, file_with_key):
    with open(file_with_key, 'w', encoding='utf-8') as f:
        for uc in use_case_list:
            if 'ucName' in uc:
                uc["key_name"] = get_key_ucname(uc['ucName'])

            if 'uctext' in uc:
                uc["key_act"] = get_key_pub(uc['uctext'], "action")
                uc["key_obj"] = get_key_pub(uc['uctext'], "entity")

            f.write(json.dumps(uc, ensure_ascii=False) + '\n')

# 3. 用字典保存edge_dict,仿照create_edges_without_ucText()函数。 用于llm as enhancer（act/obj没有按照step存放）
def create_edges_llm(use_case_list, error_word_list, uc_list_ori):
    edges_dict = {}
    node_to_UCText = {}  # 用于存放keyword、act、obj节点与UCText的对应关系
    # 没有keyword的文件，用于将act和obj根据所属于的step归类（比如act=[1,2,3,4],归类之后[[1,2],[3,4]]）
    for uc, uc_ori in zip(use_case_list, uc_list_ori):
        # 1. Make "keyword-keyword" edge(考虑了与前后node均相连)
        edges_dict = make_key_key_edge(uc, edges_dict)

        # 2. Add "keyword-act" edges, "keyword-obj" edges
        edges_dict = make_key_act_obj(uc, edges_dict)

        # 3. Add "act-act" edges (只考虑了与后面node相连，暂不考虑和前面node相连)
        edges_dict = make_common_edges(uc['act'], edges_dict, "act")

        # 4. Add "obj-obj" edges(只考虑了与后面node相连，暂不考虑和前面node相连)
        edges_dict = make_common_edges(uc['obj'], edges_dict, 'obj')

        # 5. Add "act-obj" edge
        edges_dict = make_act_obj_edges(uc, uc_ori, edges_dict, error_word_list)

        # 6. create (node to uctext)'s map
        node_to_UCText = create_node_uctext(node_to_UCText, uc)

    return edges_dict, node_to_UCText
def make_key_key_edge(uc, edges_dict):
    # only rgat方法中，keyword就是全部链接起来即可，不分act/obj。
    # 一个keyword node链接前后的keyword node即可。
    keyword_list = list(chain(uc['key_act'], uc['key_obj']))
    for key in ["key_name", "key_path"]:
        if key in uc:
            keyword_list = list(chain(keyword_list, uc[key]))

    # 1、创建source node(dict 的 key)
    for item in keyword_list:
        if (item, 'keyword') not in edges_dict:
            edges_dict[(item, 'keyword')] = []
    # 2、遍历列表，为每个node找到相邻的node
    for i in range(len(keyword_list)):
        # if i > 0:  # 因为后面变成无向边了（有向边反转），所以不需要添加前一个元素到当前元素了
        #     # 添加前一个元素到当前元素的相邻列表中
        #     edges_dict[(keyword_list[i], 'keyword')].append((keyword_list[i - 1], 'keyword'))
        if i < len(keyword_list) - 1:
            # 添加后一个元素到当前元素的相邻列表中
            edges_dict[(keyword_list[i], 'keyword')].append((keyword_list[i + 1], 'keyword'))

    return edges_dict

# 统计节点数量（入参为dict）, 根据子图统计
def count_node_data_dict_sub(edges_dict_list):
    node_total, node_max_sub_data = 0, 0
    act_node_num, obj_node_num, key_node_num = [], [], []  # 不同子图会包含相同的点，所以需要去重
    edge_num_total = 0
    edges_deduplicate_all = []
    for edges_list in edges_dict_list:
        print(f'edge_list index: {edges_dict_list.index(edges_list)}')
        # 1. 计算每个子图中各种点的个数 (这里去重的edge数没意义（edges_deduplicate），因为只是每个子图的去重，整体还是会有重复)
        act, obj, key, edges_num, edges_deduplicate = count_node_sub_data(edges_list)

        # 2. 记录所有子图中最多node(act+obj+key)数目
        if (len(act) + len(obj) + len(key)) > node_max_sub_data:
            node_max_sub_data = (len(act) + len(obj) + len(key))

        # 3. 统计各个类别的node总数
        act_node_num.extend(set(act) - set(act_node_num))
        obj_node_num.extend(set(obj) - set(obj_node_num))
        key_node_num.extend(set(key) - set(key_node_num))

        # # 4. 累加 edge 总数
        # edge_num_total += edges_num
        #
        # # 5. edge 去重的总数
        # edges_deduplicate_all = find_diff_list(edges_deduplicate_all, edges_deduplicate)

    # 6. 累加所有node(act+obj+key)数目
    node_total = len(act_node_num) + len(obj_node_num) + len(key_node_num)

    print(
        f'node总数: {node_total}, act总数: {len(act_node_num)}, obj总数: {len(obj_node_num)},'
        f' key总数: {len(key_node_num)}, 子图中最多node数: {node_max_sub_data}')
    # f' edge总数: {edge_num_total}, edge 去重后总数：{len(edges_deduplicate_all)}.')

    return {'node_total': node_total, 'act_node': len(act_node_num), 'obj_node': len(obj_node_num),
            'key_node': len(key_node_num),
            'node_max_sub': node_max_sub_data, 'edge_num_total': edge_num_total}
    # 'edge_dedup': len(edges_deduplicate_all)


# 制作所有"keyword-act" edges, "keyword-obj" edges。其中act/obj 没有按照step 存放
def make_key_act_obj(uc, edges_dict):
    # 先不考虑去重，先把边都加上（不去重也可以帮助后续数）
    keyword_list = []
    for key in ["key_name", "key_path"]:
        if key in uc:
            keyword_list = list(chain(keyword_list, uc[key]))

    if 'key_act' in uc.keys():
        keyword_list = list(chain(uc['key_act'], uc['key_obj'], keyword_list))

    for keyword in keyword_list:
        if is_nested_list(uc["act"]) or is_nested_list(uc["obj"]):  # 如果是已经按照step存放，即是嵌套列表
            if 'steps' in uc.keys():
                act_list, obj_list = process_none_in_gt_node(uc['act'], uc['obj'], uc['steps'])
                act_list = flatten_list(act_list)
                obj_list = flatten_list(obj_list)
            else:
                act_list = flatten_list(uc['act'])
                obj_list = flatten_list(uc['obj'])

        for act in act_list:
            edges_dict[(keyword, 'keyword')].append((act, 'act'))
        for obj in obj_list:
            edges_dict[(keyword, 'keyword')].append((obj, 'obj'))

    return edges_dict


def is_nested_list(input_list):
    if not isinstance(input_list, list):
        return False

    for element in input_list:
        if isinstance(element, list):
            return True
    return False


# 制作act-act 或者 obj-obj 的边（PUB DATASET用这个,其他也可能用到）
def make_common_edges(node_list, edges_dict, label):
    if is_nested_list(node_list):  # 如果是嵌套list，先把list展开
        node_list = flatten_list(node_list)
    for node in node_list:
        if (node, label) not in edges_dict.keys():
            edges_dict[(node, label)] = []

    # 只链接node后一个act/obj即可，不管前一个(与keyword不同，感觉应该考虑一下是否应该考虑前一个node)
    for i in range(len(node_list)):
        if i < len(node_list) - 1:
            # 添加后一个元素到当前元素的相邻列表中
            edges_dict[(node_list[i], label)].append((node_list[i + 1], label))

    return edges_dict


def create_node_uctext(node_to_UCText, uc):
    keyword_list = list(chain(uc['key_act'], uc['key_obj']))
    for key in ["key_name", "key_path"]:
        if key in uc:
            keyword_list = list(chain(keyword_list, uc[key]))

    for keyword in keyword_list:
        if f"{keyword}_{'keyword'}" not in node_to_UCText.keys():
            node_to_UCText[f"{keyword}_{'keyword'}"] = []
        node_to_UCText[f"{keyword}_{'keyword'}"].append(uc['index'])

    for act_list in uc['act']:
        for act in act_list:
            if f"{act}_{'act'}" not in node_to_UCText.keys():
                node_to_UCText[f"{act}_{'act'}"] = []
            node_to_UCText[f"{act}_{'act'}"].append(uc['index'])

    for obj_list in uc['obj']:
        for obj in obj_list:
            if f"{obj}_{'obj'}" not in node_to_UCText.keys():
                node_to_UCText[f"{obj}_{'obj'}"] = []
            node_to_UCText[f"{obj}_{'obj'}"].append(uc['index'])

    return node_to_UCText

# 生成数据集,修改自_6_multi_Data_makeDataset 的 generate_dataset_1()
# 这个为多数据集版本，且包含x, edge_index, edge_type, edge_attr、y、train_mask
def generate_dataset_4turbo(edges_dict, node_to_UCText, para_dict):
    # 1、获得边的重复度，之后可用作权重
    for key in edges_dict.keys():
        edges_dict[key] = [(item, count) for item, count in Counter(edges_dict[key]).items()]

    # 2、节点到编号的映射(编号是当前子图中节点的编号，从0开始)
    node_to_id = {}
    id_to_node = {}

    # 遍历边数据以建立节点到编号的映射和收集节点类型
    next_id = 0
    for source_node, edges in edges_dict.items():
        source_node_str = f"{source_node[0]}_{source_node[1]}"  # 节点名称+类型作为唯一标识，因为可能用名称相同，但是类型不同的节点。如果仅用名称为标识，则可能出错
        if source_node_str not in node_to_id:
            node_to_id[source_node_str] = next_id
            id_to_node[next_id] = source_node_str
            next_id += 1
        for target_node, times in edges:
            target_node_str = f"{target_node[0]}_{target_node[1]}"
            if target_node_str not in node_to_id:
                node_to_id[target_node_str] = next_id
                id_to_node[next_id] = target_node_str
                next_id += 1

    # 3、创建节点特征embedding
    node_embeddings = []
    for node_id in id_to_node:
        node_str = id_to_node[node_id]
        # 分割节点名称和类型
        node_name, node_type = node_str.rsplit('_', 1)
        # 获取嵌入
        embedding = MODEL.encode([node_name])[0]
        embedding = torch.from_numpy(embedding)  # 将这个 NumPy 数组转换为 PyTorch 张量
        node_embeddings.append(embedding)

    # 将嵌入转换为PyTorch张量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.stack(node_embeddings, dim=0).to(device, dtype=torch.float)

    # 4、创建边数据
    edge_index = []
    edge_attr = []  # 边权重
    edge_type = []  # 用于存储边的类型

    for source_node, edges in edges_dict.items():
        source_id = node_to_id[f"{source_node[0]}_{source_node[1]}"]
        for target_node, weight in edges:
            target_id = node_to_id[f"{target_node[0]}_{target_node[1]}"]
            edge_index.append([source_id, target_id])
            edge_attr.append(weight)

            # 确定边的类型
            if source_node[1] == 'act' and target_node[1] == 'act':
                edge_type.append(0)  # 假设0表示act-act边
            elif source_node[1] == 'act' and target_node[1] == 'obj':
                edge_type.append(1)  # 假设1表示act-obj边
            elif source_node[1] == 'obj' and target_node[1] == 'obj':
                edge_type.append(2)  # 假设2表示obj-obj边
            elif source_node[1] == 'keyword' and target_node[1] == 'act':
                edge_type.append(3)  # 假设3表示 keyword-act 边
            elif source_node[1] == 'keyword' and target_node[1] == 'obj':
                edge_type.append(4)  # 假设4表示 keyword-obj 边
            elif source_node[1] == 'keyword' and target_node[1] == 'keyword':
                edge_type.append(5)  # 假设5表示 keyword-keyword 边
            else:
                print(source_node[1], target_node[1])
                print(f'edge_type Error!!!')
                print(source_node[1], target_node[1])

    # 将edge_index和edge_type转换为PyTorch张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index_reverse = edge_index[[1, 0], :]  # 将edge的index反转，等于是将原本有向边变为无向边，因为两个方向的edge都有了
    edge_index = torch.cat((edge_index, edge_index_reverse), dim=1)  # 然后直接拼接在后面

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # 权重是出现次数，所以使用int
    edge_attr = edge_attr.view(-1, 1)  # 转换为二维，形状为 [num_edges, 1]
    edge_attr = torch.cat((edge_attr, edge_attr), dim=0)  # 有向边变成无向边，权重直接复制一份放在后面即可

    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_type = torch.cat((edge_type, edge_type), dim=0)  # 有向边变成无向边，类型也是直接复制一份放在后面即可

    # 5、生成非预定义参数（不能用y的原因是Data对象要求y必须是[节点数]的向量，形状不能变化）
    # 5.1、生成标签 y，是一个列表，顺序对应keyword，存放每个keyword关联的act/obj的id
    y = []  # 顺序按照node_to_id中keyword的顺序
    keyword_id = []  # 记录keyword节点的id
    for key_node in node_to_id:
        if 'keyword' in key_node:
            keyword_id.append(node_to_id[key_node])
            related_node = []
            node_name, node_type = key_node.rsplit('_', 1)
            for (tar_node_str, target_node_type), time in edges_dict[(node_name, node_type)]:
                if target_node_type != 'keyword':
                    tar_node = f"{tar_node_str}_{target_node_type}"
                    related_node.append(node_to_id[tar_node])
            y.append(related_node)
        else:
            y.append([-1])  # 除了keyword之外的节点位置补充-1

    # 确定最大的列表长度为 节点总数,需要所有列表长度一致
    max_length = para_dict['max_node_subdata']  # 所有子图最多节点数原为1257
    # 创建一个填充后的二维列表，填充'-99'表示无效数据，补齐列表
    padded_y = [sublist + [-99] * (max_length - len(sublist)) for sublist in y]
    # 将填充后的列表转换为张量
    y_tensor = torch.tensor(padded_y, dtype=torch.long)

    # # 5.2、生成训练掩码train_mask, 用整张子图（Data对象）来训练，因为抽百分比的数据的话，数据集会不平衡（各种类型的点数量不一致）
    # # 计算训练节点的数量
    # train_mask = torch.zeros(len(id_to_node), dtype=torch.bool)
    # train_mask[keyword_id] = True

    # 6、转变node_to_UCText为非预定义参数
    node_to_UCText_new = {}  # 创建一个新的用于存放
    for node in node_to_UCText.keys():
        node_id = node_to_id[node]
        node_to_UCText_new[node_id] = node_to_UCText[node]  # node_to_UCText_new中item是 节点id：uctext id
        node_to_UCText_new[node_id] = list(set(node_to_UCText_new[node_id]))  # 列表去重

    sort_key = sorted(node_to_UCText_new.keys())  # 按照节点id排序
    padded_UC = [[item + para_dict['uctext_start'] for item in node_to_UCText_new[key]] for key in
                 sort_key]  # 按照节点id排序 并从总节点id (原来为)3241后开始计算UCText的id

    # 填充无效数据'-99'至最长维度，这里的最长长度max_length变成了众多子图中最多的UC数目
    max_length = para_dict['max_length']  # 原为1744
    # 遍历result中的每个小列表
    for i in range(len(padded_UC)):
        # 计算当前小列表需要的填充长度
        padding_length = max_length - len(padded_UC[i])
        # 如果需要填充，则用-99填充到长度为max_length
        if padding_length > 0:
            padded_UC[i] += [-99] * padding_length

    for sublist in padded_UC:  # 检查长度是否补齐
        if len(sublist) != max_length:
            print(f"子列表 {sublist} 的长度不等于最长{max_length}，其id为：{padded_UC.index(sublist)}")
    node_to_UCText_new = torch.tensor(padded_UC, dtype=torch.long)

    # 7、将数据添加到Data对象中,包含x、edge_index、edge_attr和edge_attr
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)
    data.keyword_relation = y_tensor
    data.to_UCText = node_to_UCText_new  # 表示 每个node存在于哪个UCText里，节点id：UCText_id

    return data


if __name__ == "__main__":
    NECT_groud_truth_path = "/20240421/ControlledExper/2_dataset_origin_node/Ernie_NEC-T_groud_truth.json"
    NECT_groud_truth_reextract = "E:/bertTest/20240421/ControlledExperiment_conventionAlgorithm/2_dataset_origin_node/Ernie_NEC-T_groud_truth_reextract.json"
    NECT_groud_truth_with_keyword = "E:/bertTest/20240421/ControlledExperiment_conventionAlgorithm/2_dataset_origin_node/Ernie_NEC-T_groud_truth_with_keyword.json"

    print(f' 程序开始时间：{datetime.now().strftime("%H:%M:%S")}')
    # 确认要完成的 task_index
    task_index = 2  # 选择需要完成的task
    
    # task2: (NCE-T，pt文件)llm as enhancer 的制作pt文件，专用于中文、NCET数据集（按照_6_multi_Data_makeDataset.py写）
    if task_index == 2:
        origin_file = "data/NCE-T_DATASET/NCE-T_tc.json"
        file_with_key = "ControlledExper/2_dataset_origin_node/Ernie-4-Turbo/with_keyword/Ernie_NCET_ground_truth.json"
        pt_save_path = "ControlledExper/5_experiment_data/2_pt_file/chatgpt/1st_before_formalized/NCE_T_llm_20.pt"
        file_no_key = 'ControlledExper/2_dataset_origin_node/Ernie-4-Turbo/Ernie_NCET_ground_truth.json'

        # 1.读取数据
        use_case_list_dict = read_uc_from_json(file_with_key)
        use_case_list_dict = format_key_node(use_case_list_dict)  # 此时key node是str格式，改成list

        # 1.1 Delete error node
        for uc in use_case_list_dict:
            uc = delete_error_node_4turbo(uc, ERROR_WORD_LIST)
        # 1.2 生成全局的边数目和node数目
        argc1 = check_argc(use_case_list_dict)

        # # 2. 由于数据集太大，需要将数据分组。现按照path分组。group_uc()函数为随机分配
        # use_case_list_ori = extract_use_case_step_1(origin_file)
        # use_case_list = group_uc(use_case_list_ori, use_case_list_dict)

        # 2. 因为随机分组时间太久，所以选择保存一份.group_uc_fixed()为固定分组
        use_case_list = group_uc_fixed(use_case_list_dict, GROUPING_UC_20)
        # 2.1 no keyword文件用于将act和obj根据所属于的step归类（比如act=[1,2,3,4],归类之后[[1,2],[3,4]]），所以需要相同分类
        use_case_list_ori = group_uc_fixed(read_uc_from_json(file_no_key), GROUPING_UC_20)

        # 3. 获得两个字典：edges_dict(存放act/obj/keyword之间的edge)；node_to_UCText_list(存放act/obj/keyword to uctext的映射)
        edges_dict_list, node_to_UCText_list = get_edges_dict_and_node_to_UCText_4turbo(use_case_list, ERROR_WORD_LIST,
                                                                                        use_case_list_ori)

        # 3.1 统计：所有子图中最多的节点数（act+obj+key）
        argc2 = count_node_data_dict_sub(edges_dict_list)
        # 3.2 判断两次统计是否一致(argc2本来也比argc1多几项)
        print(f'全局统计和分图统计不一致的项为：{find_diff_dict(argc1, argc2)}')

        # 4、生成数据集
        dataset = []
        # uctext_start 为uctext节点的起始点（全部数据中act+obj+key节点总数）；max_node_subdata 所有子图最多节点数（act+obj+key）; max_length 所有子图中包含最多uc的个数
        para_dict = {"uctext_start": argc2['node_total'], "max_node_subdata": argc2['node_max_sub'],
                     "max_length": ARGC_20['max_uc_in_sub']}
        for i in range(len(edges_dict_list)):
            subgraph_data = generate_dataset_4turbo(edges_dict_list[i], node_to_UCText_list[i], para_dict)
            dataset.append(subgraph_data)
        # 5、将data保存到文件
        torch.save(dataset, pt_save_path)