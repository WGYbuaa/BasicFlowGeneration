from RGAT_model import *
import torch_geometric
import json
from torch.optim import lr_scheduler
from datetime import datetime
import os

num_epochs = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pt_only_ragt_path = '/root/autodl-tmp/ASSAM/program/data/5_experiment_data/2_pt_file/1_only_rgat/'
ernie_llm_as_enhancer_path = '/root/autodl-tmp/ASSAM/program/data/5_experiment_data/2_pt_file/2_llm/ernie/pub_Ernie_20_1.pt'
chatgpt_llm_as_enhancer_path = '/root/autodl-tmp/ASSAM/program/data/5_experiment_data/2_pt_file/2_llm/chatgpt/pub_chatgpt_20_1.pt'
gpt_as_enhancer_ncet_path = '/root/autodl-tmp/ASSAM/program/data/5_experiment_data/2_pt_file/2_llm/chatgpt/NCE_T_llm_20.pt'
robust_path = '/root/autodl-tmp/ASSAM/program/data/5_experiment_data/2_pt_file/2_llm/chatgpt/robust_exper/90_NCE_T_llm.pt'

# 0、加载数据
# 只有当torch_geometric.loader.DataLoader第一个参数列表中Data对象多于batch_size时，后续循环才能减少输入的数据量，
# 其他情况（batch_size小于等于Data对象数目），都只会全部的Data数据输入循环，占用的GPU内存也不会减少，设置批次不起作用。

lambda_ls = [0.5, 0.55, 0.6]
robust_ls = ["80_NCE_T"]
for gamma in lambda_ls:
    for rob in robust_ls:
        rob_path = robust_path.replace("90_NCE_T", rob)
        for i in range(3):
            # load 工业数据集
            loader = torch.load(gpt_as_enhancer_ncet_path)
            dataset_train = torch_geometric.loader.DataLoader(loader[:338], batch_size=1, shuffle=False)  
            dataset_eval = torch_geometric.loader.DataLoader(loader[338:370], batch_size=1, shuffle=False)   
            dataset_test = torch_geometric.loader.DataLoader(loader[370:], batch_size=1, shuffle=False)

            # # load pub数据集
            # loader = torch.load(chatgpt_llm_as_enhancer_path)
            # dataset_train = torch_geometric.loader.DataLoader(loader[:30], batch_size=1, shuffle=True)  # 40: 18 18:20 20
            # dataset_eval = torch_geometric.loader.DataLoader(loader[30:33], batch_size=1, shuffle=True)   # 20: 30  30:33 33
            # dataset_test = torch_geometric.loader.DataLoader(loader[33:], batch_size=1, shuffle=True)
            # 1、初始化模型
            edge_dim = 1  # 获取边特征的维度
            model = RGAT(num_edge_types=6, in_channels=512,
                        out_channels=64, edge_dim=edge_dim, gamma = gamma)

            # 2、定义优化器
            learning_rate = 1e-4 # 1e-4/5e-5
            optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)  # 调学习率

            # 3、移动模型到GPU
            model = model.to(device)

            # 4、训练循环。打开一个文件用于写入los
            output_for_json = []  # 存放metric数据
            loss_data_json = []  # 存放loss数据
            print("NOW : ")
            print(i)
            best_AUC = 0.
            best_epoch = 0

            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for data in dataset_train:
                    optimizer.zero_grad()
                    # 1、部分数据放在GPU上
                    x = data.x.to(device)
                    edge_index = data.edge_index.to(device)
                    edge_type = data.edge_type.to(device)
                    edge_attr = data.edge_attr.to(device)
                    # print(f"number of UCTetx :{data.to_UCText.size()}")
                    # continue

                    # 2、训练得到模型对于每个节点的表示
                    optimizer.zero_grad()
                    out = model(x, edge_index, edge_type, edge_attr)

                    # 3-5、得到pred和ground-truth的两个张量
                    uctext_act_obj_pred, uctext_act_obj_matrix_ground_truth = get_two_matrix(data, out, device)

                    # 6、计算二分类交叉熵BCELoss
                    bce_loss = nn.BCELoss()
                    loss = bce_loss(uctext_act_obj_pred, uctext_act_obj_matrix_ground_truth.float())

                    # 梯度清空
                    loss.backward()
                    optimizer.step()

                # 7、记录loss
                running_loss += loss.item()
                if (epoch + 1) % 5 == 0:
                    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
                    loss_data_json.append({'Epoch': (epoch + 1)/num_epochs, 'Loss': loss.item()})

                # 7.1、打印并保存整个epoch的平均loss
                epoch_loss = running_loss / len(dataset_train)
                print(f'Epoch {epoch + 1}, Average Loss: {epoch_loss}')
                loss_data_json.append({'Epoch': epoch + 1, 'Average Loss': epoch_loss})

                # 8、每个epoch计算一次
                model.eval()
                with torch.no_grad():
                    TP, TN, FP, FN = 0, 0, 0, 0
                    output_for_json.append(f'EVAL Epoch: {epoch + 1}')
                    ls_pred, ls_target = [], []
                    for data in dataset_eval:
                        data = data.to(device)
                        # print("EVAL DATA : " + str(data.x.shape))
                        out = model(data.x, data.edge_index, data.edge_type, data.edge_attr)
                        preds, target = get_two_matrix(data, out, device)

                        preds = preds.view(1, -1)
                        target = target.view(1, -1)

                        ls_pred.append(preds)
                        ls_target.append(target)

                        pred_labels = (preds >= 0.5).float()

                        # 计算 TP, TN, FP, FN
                        TP += ((pred_labels == 1) & (target == 1)).sum().item()
                        TN += ((pred_labels == 0) & (target == 0)).sum().item()
                        FP += ((pred_labels == 1) & (target == 0)).sum().item()
                        FN += ((pred_labels == 0) & (target == 1)).sum().item()


                    preds = torch.cat(ls_pred, dim = 1)
                    target = torch.cat(ls_target, dim = 1)
                    output = metric_collection(preds, target)

                    # print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
                    # 计算精确率（Precision）
                    if TP + FP == 0:
                        avg_p = 0
                    else:
                        avg_p = TP / (TP + FP)

                    # 计算召回率（Recall）
                    if TP + FN == 0:
                        avg_r = 0
                    else:
                        avg_r = TP / (TP + FN)

                    # 计算 F1-score
                    if avg_p + avg_r == 0:
                        avg_f = 0
                    else:
                        avg_f = 2 * avg_p * avg_r / (avg_p + avg_r)

                    AUCnow = output['BinaryAUROC'].item()

                    if(AUCnow > best_AUC):
                        best_AUC = AUCnow
                        best_epoch = epoch + 1
                        print("best_AUC = {}".format(best_AUC))
                        print("best_epoch = {}".format(best_epoch))
                    if(epoch - best_epoch >= 10):
                        print("Finished Training.")
                        break

                    output_for_json.append(f'EVAL Epoch: {epoch + 1}')
                    output_for_json.append({"BinaryPrecision": avg_p})
                    output_for_json.append({"BinaryRecall": avg_r})
                    output_for_json.append({"BinaryF1Score": avg_f})
                    output_for_json.append({"BinaryAUROC": output['BinaryAUROC'].item()})

                    print(f'EVAL Epoch: {epoch + 1} ;  P = ' + str(avg_p) + " R = " + str(avg_r) + " F1 = " + str(avg_f) + " AUC = " + str(output['BinaryAUROC'].item()))

                # 9、每个epoch测试一次
                model.eval()
                with torch.no_grad():
                    TP, TN, FP, FN = 0, 0, 0, 0
                    output_for_json.append(f'TEST Epoch: {epoch + 1}')
                    ls_pred, ls_target = [], []
                    for data in dataset_test:
                        data = data.to(device)
                        out = model(data.x, data.edge_index, data.edge_type, data.edge_attr)
                        preds, target = get_two_matrix(data, out, device)

                        preds = preds.view(1, -1)
                        target = target.view(1, -1)

                        ls_pred.append(preds)
                        ls_target.append(target)

                        pred_labels = (preds >= 0.5).float()

                        # 计算 TP, TN, FP, FN
                        TP += ((pred_labels == 1) & (target == 1)).sum().item()
                        TN += ((pred_labels == 0) & (target == 0)).sum().item()
                        FP += ((pred_labels == 1) & (target == 0)).sum().item()
                        FN += ((pred_labels == 0) & (target == 1)).sum().item()


                    preds = torch.cat(ls_pred, dim = 1)
                    target = torch.cat(ls_target, dim = 1)
                    output = metric_collection(preds, target)

                    # print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
                    # 计算精确率（Precision）
                    if TP + FP == 0:
                        avg_p = 0
                    else:
                        avg_p = TP / (TP + FP)

                    # 计算召回率（Recall）
                    if TP + FN == 0:
                        avg_r = 0
                    else:
                        avg_r = TP / (TP + FN)

                    # 计算 F1-score
                    if avg_p + avg_r == 0:
                        avg_f = 0
                    else:
                        avg_f = 2 * avg_p * avg_r / (avg_p + avg_r)

                    output_for_json.append(f'TEST Epoch: {epoch + 1}')
                    output_for_json.append({"BinaryPrecision": avg_p})
                    output_for_json.append({"BinaryRecall": avg_r})
                    output_for_json.append({"BinaryF1Score": avg_f})
                    output_for_json.append({"BinaryAUROC": output['BinaryAUROC'].item()})

                    print(f'TEST Epoch: {epoch + 1} ;  P = ' + str(avg_p) + " R = " + str(avg_r) + " F1 = " + str(avg_f) + " AUC = " + str(output['BinaryAUROC'].item()))


            # 10、打印记录
            loss_file_path = '/root/autodl-tmp/ASSAM/program/data/5_experiment_data/1_result/2_llm/1124/loss.json'
            metrics_file_path = '/root/autodl-tmp/ASSAM/program/data/5_experiment_data/1_result/2_llm/1124/metrics.json'
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_date = datetime.now().strftime("%m%d")
            loss_file_path = loss_file_path.replace('1124', current_date)
            metrics_file_path = metrics_file_path.replace('1124', current_date)
            new_metrics_file_path = os.path.splitext(metrics_file_path)[0] + "_" + current_time + "_" + str(learning_rate) + "_" + str(gamma) + "_" + rob + ".json"
            new_loss_file_path = os.path.splitext(loss_file_path)[0] + "_" + current_time + "_" + str(learning_rate) + "_" + str(gamma) + "_" + rob + ".json"

            directory = os.path.dirname(new_metrics_file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = os.path.dirname(new_loss_file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            f = open(new_loss_file_path, "w", encoding = "utf-8")
            for item in loss_data_json:
                json_str = json.dumps(item)
                f.write(json_str + '\n')
            f.close()
            f = open(new_metrics_file_path, 'w', encoding = "utf-8")
            for item in output_for_json:
                json_str = json.dumps(item)
                f.write(json_str + '\n')
            f.close()



            # 9、根据num_epochs构建文件名
            file_name = f'/root/autodl-tmp/ASSAM/program/data/5_experiment_data/1_result/2_llm/1124/ep{num_epochs}_model_server_' + current_time + '.pth'
            torch.save(model.state_dict(), file_name)
