# generat .pt file for GPT on pub dataset
# The functions in this file can be found in the 0_ERNIE_pub.py file.

if __name__ == "__main__":
    NECT_groud_truth_path = "/20240421/ControlledExper/2_dataset_origin_node/Ernie_NEC-T_groud_truth.json"
    NECT_groud_truth_reextract = "E:/bertTest/20240421/ControlledExperiment_conventionAlgorithm/2_dataset_origin_node/Ernie_NEC-T_groud_truth_reextract.json"
    NECT_groud_truth_with_keyword = "E:/bertTest/20240421/ControlledExperiment_conventionAlgorithm/2_dataset_origin_node/Ernie_NEC-T_groud_truth_with_keyword.json"

    print(f' 程序开始时间：{datetime.now().strftime("%H:%M:%S")}')
    # 确认要完成的 task_index
    task_index = 2  # 选择需要完成的task

    # task7:(pub,pt,chatgpt)制作pt文件。仿照上述task6
    if task_index == 7:
        in_file = "E:/GitHub/ASSAM/data/2_dataset_origin_node/Chatgpt_4o/4rd_after_formalized/GPT_pub_gt.json"
        pt_save_path = "ControlledExper/5_experiment_data/2_pt_file/chatgpt/2nd_after_formalized/pub_gpt_20.pt"

        # 1、读取文件
        use_case_list = read_uc_from_json(in_file)

        index = 0
        # 2、将数据集名称作为 key_path。更新全局index
        for uc in use_case_list:
            uc['key_path'] = format_node_eng(uc['dataset'])
            for key in ["ucName", "pred_steps", "pred_act", "pred_obj", "tp_act", "tp_obj"]:  # 删除一些 不然内存不够
                if key in uc.keys():
                    del uc[key]
            if "key_act" not in uc.keys():
                uc['key_act'] = []
            if "key_obj" not in uc.keys():
                uc['key_obj'] = []
            uc['index'] = index
            index += 1

        # 3、 生成全局的边数目和node数目
        argc1 = check_argc(use_case_list)

        # # 4、进行分组
        # use_case_list = group_pub_uc(use_case_list)
        # 4. 因为随机分组时间太久，所以选择保存一份.group_uc_fixed()为固定分组
        use_case_list = group_uc_fixed(use_case_list, PUB_GROUPING_UC_20_1)

        # 5. 获得两个字典：edges_dict(存放act/obj/keyword之间的edge)；node_to_UCText_list(存放act/obj/keyword to uctext的映射)
        edges_dict_list, node_to_UCText_list = get_edges_dict_and_node_to_UCText_only_rgat(use_case_list)

        # 5.1 统计：所有子图中最多的节点数（act+obj+key）
        argc2 = count_node_data_dict_sub(edges_dict_list)
        # 5.2 判断两次统计是否一致. 除了'node_max_sub'和’edge_num_total‘是后面加的，其他都应该一样
        print(f'全局统计和分图统计不一致的项为：{find_diff_dict(argc1, argc2)}')

        # 6、生成数据集
        dataset = []
        # uctext_start 为uctext节点的起始点（全部数据中act+obj+key节点总数）；max_node_subdata 所有子图最多节点数（act+obj+key）; max_length 所有子图中包含最多uc的个数
        para_dict = {"uctext_start": argc2['node_total'], "max_node_subdata": argc2['node_max_sub'],
                     "max_length": ARGC_20['max_uc_in_sub']}
        for i in range(len(edges_dict_list)):
            subgraph_data = generate_dataset_4turbo(edges_dict_list[i], node_to_UCText_list[i], para_dict)
            dataset.append(subgraph_data)
        # 7、将data保存到文件
        torch.save(dataset, pt_save_path)