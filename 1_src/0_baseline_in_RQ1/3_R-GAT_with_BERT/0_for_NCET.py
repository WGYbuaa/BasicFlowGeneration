# only RGAT is RGAT with BERT

def is_valid_path(path):
    try:
        # 检查路径是否存在
        if not os.path.exists(path):
            return False

        # 检查路径是否为文件
        if os.path.isfile(path):
            # 获取文件大小（以字节为单位）
            file_size = os.path.getsize(path)
            # 检查文件大小是否大于0
            return file_size > 0
        else:
            # 如果路径不是文件（例如目录），则认为无效
            return False
    except Exception as e:
        # 捕获任何异常并返回False
        print(f"An error occurred: {e}")
        return False


def extract_use_case_step_1(path):
    use_case_list = list()  # 按顺序存放所有的用例uc
    use_case = useCase()
    uc_number = 0  # uc的全局index
    step_number = 0  # step在uc内的位置

    path = path.replace('\\', '/')
    with open(path, encoding='utf-8') as f:
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

                    # if len(use_case_list) == 10:  # 先用前三十个use case 做demo。不用的时候注释掉这两行即可
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

                # 提取动作-数据对象
                action_object = extract_action_object(use_case_step.step)

                use_case_step.action = action_object[0]
                use_case_step.object.append(action_object[1])

                if len(use_case_step.object[0]) == 0 and len(use_case_step.action) != 0:
                    use_case_step.object[0] = use_case_step.action  # 字典模型中许多ROOT其实为obj，但被放在了action

                if len(use_case_step.action) == 0 or len(use_case_step.object[0]) == 0:
                    print(use_case_step.action, use_case_step.object, use_case_step.index_global)

                # 后面判断重复度基本都使用.step_clean，这样可以删除不必要的定状补。修饰过多，会提高原本不同的act+obj的相似度
                use_case_step.step_clean = use_case_step.action + str(use_case_step.object[0])



    return use_case_list

def turn_class_to_dict(use_case_list):
    update_list = []
    for uc in use_case_list:
        update_dict = {'index': 0, 'steps': [], 'act': [], 'obj': [], 'dataset': 'NCE-T', 'ucText': '', 'ucPath': '',
                       'key_act': [], 'key_obj': [], 'key_path': []}
        update_dict['index'] = uc.index_global
        update_dict['steps'] = uc.step_former
        update_dict['ucText'] = uc.tc_text
        update_dict['ucPath'] = uc.tc_path
        for step in uc.step_list:
            update_dict['act'].append([step.action])
            update_dict['obj'].append(step.object)

        update_list.append(update_dict)
    return update_list

# 提取uctext\path中关键词，用于 only rgat
def extract_keywords_dict(use_case_list):
    for uc in use_case_list:
        # 1、提取tc_text中的关键词
        text = uc['ucText']
        # 假设所有名词和动词都是关键词
        uc['key_act'] = [word for word, flag in jieba.posseg.cut(text) if flag in ['v']]
        uc['key_obj'] = [word for word, flag in jieba.posseg.cut(text) if flag in ['n']]

        # 2、提取tc_path中的名词和动词
        # 使用split方法按照"/"分割字符串
        parts = uc['ucPath'].split('/')
        # 使用列表推导式和正则表达式去除每个部分中的数字和下划线
        uc['key_path'] = [re.sub(r'\d+_', '', part) for part in parts]

    return use_case_list

def write_to_json(outfile_path, use_cases_list):
    with open(outfile_path, 'w', encoding='utf-8') as f:
        for uc in use_cases_list:
            f.write(json.dumps(uc, ensure_ascii=False) + '\n')



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

def get_edges_dict_and_node_to_UCText_only_rgat(use_case_list):
    edges_dict_list = []
    node_to_UCText_list = []
    for uc_list in use_case_list:
        edge_dict, node_to_UCText = create_edges_llm_only_rgat(uc_list)
        edges_dict_list.append(edge_dict)
        node_to_UCText_list.append(node_to_UCText)

    return edges_dict_list, node_to_UCText_list

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

if __name__ == '__main__':

    task_index = 2

    # only rgat 的 NCET数据集的 pt 文件的制作过程，之前那个NCE_T_dataset_muti_data.py 不用了
    if task_index == 2:
        print("only rgat 的 pt 文件的制作过程，之前那个NCE_T_dataset_muti_data.py 不用了")
        origin_file = "data/NCE-T_DATASET/NCE-T_tc.json"
        file_with_key = "ControlledExper/2_dataset_origin_node/only_rgat/with_key/only_rgat_NCET_ground_truth.json"
        pt_save_path = "ControlledExper/5_experiment_data/2_pt_file/only_rgat/1st_before_formalized/NCE_T_only_rgat.pt"

        # 1. 读取 use_case_list 数据
        if not is_valid_path(file_with_key):  # 如果已经跑出来了dict格式的uc数据
            # 1.1 将uc数据从原始数据提取出来，获得keyword等数据后，存入 only_rgat_NCET_ground_truth.json 中。
            # 1.1.1、从.json文件中提取出use_case_list
            use_case_list = extract_use_case_step_1(origin_file)
            # 1.1.2、将 use_case_list 从class转换为dict
            use_case_list = turn_class_to_dict(use_case_list)
            # 1.1.3、加入关键词
            use_case_list = extract_keywords_dict(use_case_list)
            # 1.1.4、保存一下
            write_to_json(file_with_key, use_case_list)
        else:
            # 1.1、从.json文件中提取出use_case_list
            use_case_list = read_uc_from_json(file_with_key)
            # 1.2 生成全局的边数目和node数目
            argc1 = check_argc(use_case_list)

        # 2. 因为随机分组时间太久，所以选择保存一份.group_uc_fixed()为固定分组
        use_case_list = group_uc_fixed(use_case_list, GROUPING_UC_20)

        # 3. 获得两个字典：edges_dict(存放act/obj/keyword之间的edge)；node_to_UCText_list(存放act/obj/keyword to uctext的映射)
        edges_dict_list, node_to_UCText_list = get_edges_dict_and_node_to_UCText_only_rgat(use_case_list)

        # 3.1 统计：所有子图中最多的节点数（act+obj+key）
        argc2 = count_node_data_dict_sub(edges_dict_list)
        # 3.2 判断两次统计是否一致
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

    print("******** finish ********")