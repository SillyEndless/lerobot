#!/usr/bin/env python3
"""
函数调用关系图可视化脚本
展示SmolVLA训练和推理过程中的函数调用关系
"""

import graphviz

def create_training_function_call_graph():
    """创建训练函数调用关系图"""
    
    # 创建有向图
    dot = graphviz.Digraph('Training_Function_Call_Graph', 
                          comment='SmolVLA训练函数调用关系图',
                          format='png')
    
    # 设置图形属性
    dot.attr(rankdir='TB', 
             size='16,20',
             dpi='300',
             fontname='SimHei',
             fontsize='16')
    
    # 设置节点默认属性
    dot.node_attr.update(shape='box',
                         style='filled',
                         fontname='SimHei',
                         fontsize='10',
                         height='0.4',
                         width='2.5')
    
    # 设置边默认属性
    dot.edge_attr.update(fontname='SimHei',
                         fontsize='8',
                         arrowsize='0.8')
    
    # 主训练入口
    with dot.subgraph(name='cluster_main_training') as main:
        main.attr(label='主训练入口', 
                 style='filled',
                 color='lightblue',
                 fontname='SimHei',
                 fontsize='14')
        
        main.node('train_loop', '训练循环\n(train_loop)', 
                 fillcolor='lightblue',
                 color='darkblue')
        main.node('forward_pass', 'SmolVLAPolicy.forward()\n前向传播', 
                 fillcolor='lightcyan',
                 color='darkblue')
    
    # 数据预处理
    with dot.subgraph(name='cluster_data_preprocessing') as data:
        data.attr(label='数据预处理', 
                 style='filled',
                 color='lightgreen',
                 fontname='SimHei',
                 fontsize='14')
        
        data.node('_prepare_batch', 'SmolVLAPolicy._prepare_batch()\n批次预处理', 
                 fillcolor='lightgreen',
                 color='darkgreen')
        data.node('normalize_inputs', 'Normalize.normalize_inputs()\n输入标准化', 
                 fillcolor='lightgreen',
                 color='darkgreen')
        data.node('normalize_targets', 'Normalize.normalize_targets()\n目标标准化', 
                 fillcolor='lightgreen',
                 color='darkgreen')
        data.node('adapt_to_pi_aloha', 'SmolVLAPolicy._pi_aloha_decode_state()\nALOHA状态适配', 
                 fillcolor='lightgreen',
                 color='darkgreen')
    
    # 特征提取
    with dot.subgraph(name='cluster_feature_extraction') as features:
        features.attr(label='特征提取', 
                     style='filled',
                     color='lightyellow',
                     fontname='SimHei',
                     fontsize='14')
        
        features.node('prepare_images', 'SmolVLAPolicy.prepare_images()\n图像预处理', 
                     fillcolor='lightyellow',
                     color='darkorange')
        features.node('prepare_language', 'SmolVLAPolicy.prepare_language()\n文本token化', 
                     fillcolor='lightyellow',
                     color='darkorange')
        features.node('prepare_state', 'SmolVLAPolicy.prepare_state()\n状态向量化', 
                     fillcolor='lightyellow',
                     color='darkorange')
        features.node('prepare_action', 'SmolVLAPolicy.prepare_action()\n动作向量化', 
                     fillcolor='lightyellow',
                     color='darkorange')
    
    # Flow Matching核心
    with dot.subgraph(name='cluster_flow_matching') as flow:
        flow.attr(label='Flow Matching核心', 
                 style='filled',
                 color='lightpink',
                 fontname='SimHei',
                 fontsize='14')
        
        flow.node('vla_forward', 'VLAFlowMatching.forward()\nFlow Matching前向', 
                 fillcolor='lightpink',
                 color='darkred')
        flow.node('sample_noise', 'VLAFlowMatching.sample_noise()\n采样噪声', 
                 fillcolor='lightpink',
                 color='darkred')
        flow.node('sample_time', 'VLAFlowMatching.sample_time()\n采样时间', 
                 fillcolor='lightpink',
                 color='darkred')
        flow.node('embed_prefix', 'VLAFlowMatching.embed_prefix()\n前缀嵌入', 
                 fillcolor='lightpink',
                 color='darkred')
        flow.node('embed_suffix', 'VLAFlowMatching.embed_suffix()\n后缀嵌入', 
                 fillcolor='lightpink',
                 color='darkred')
    
    # VLM和专家网络
    with dot.subgraph(name='cluster_vlm_expert') as vlm:
        vlm.attr(label='VLM和专家网络', 
                style='filled',
                color='lightcoral',
                fontname='SimHei',
                fontsize='14')
        
        vlm.node('vlm_forward', 'SmolVLMWithExpertModel.forward()\nVLM+专家前向', 
                fillcolor='lightcoral',
                color='darkred')
        vlm.node('embed_image', 'SmolVLMWithExpertModel.embed_image()\n图像嵌入', 
                fillcolor='lightcoral',
                color='darkred')
        vlm.node('embed_language_tokens', 'SmolVLMWithExpertModel.embed_language_tokens()\n文本嵌入', 
                fillcolor='lightcoral',
                color='darkred')
        vlm.node('forward_attn_layer', 'SmolVLMWithExpertModel.forward_attn_layer()\n自注意力层', 
                fillcolor='lightcoral',
                color='darkred')
        vlm.node('forward_cross_attn_layer', 'SmolVLMWithExpertModel.forward_cross_attn_layer()\n交叉注意力层', 
                fillcolor='lightcoral',
                color='darkred')
    
    # 损失计算
    with dot.subgraph(name='cluster_loss_computation') as loss:
        loss.attr(label='损失计算', 
                 style='filled',
                 color='lightsteelblue',
                 fontname='SimHei',
                 fontsize='14')
        
        loss.node('mse_loss', 'F.mse_loss(u_t, v_t)\nMSE损失计算', 
                 fillcolor='lightsteelblue',
                 color='darkblue')
        loss.node('loss_mean', 'loss.mean()\n平均损失', 
                 fillcolor='lightsteelblue',
                 color='darkblue')
        loss.node('backward', 'loss.backward()\n反向传播', 
                 fillcolor='lightsteelblue',
                 color='darkblue')
        loss.node('optimizer_step', 'optimizer.step()\n参数更新', 
                 fillcolor='lightsteelblue',
                 color='darkblue')
    
    # 添加边连接
    # 主训练流程
    dot.edge('train_loop', 'forward_pass')
    dot.edge('forward_pass', '_prepare_batch')
    dot.edge('forward_pass', 'vla_forward')
    
    # 数据预处理流程
    dot.edge('_prepare_batch', 'normalize_inputs')
    dot.edge('_prepare_batch', 'normalize_targets')
    dot.edge('_prepare_batch', 'adapt_to_pi_aloha')
    
    # 特征提取流程
    dot.edge('forward_pass', 'prepare_images')
    dot.edge('forward_pass', 'prepare_language')
    dot.edge('forward_pass', 'prepare_state')
    dot.edge('forward_pass', 'prepare_action')
    
    # Flow Matching流程
    dot.edge('vla_forward', 'sample_noise')
    dot.edge('vla_forward', 'sample_time')
    dot.edge('vla_forward', 'embed_prefix')
    dot.edge('vla_forward', 'embed_suffix')
    
    # VLM和专家网络流程
    dot.edge('embed_prefix', 'vlm_forward')
    dot.edge('embed_suffix', 'vlm_forward')
    dot.edge('vlm_forward', 'embed_image')
    dot.edge('vlm_forward', 'embed_language_tokens')
    dot.edge('vlm_forward', 'forward_attn_layer')
    dot.edge('vlm_forward', 'forward_cross_attn_layer')
    
    # 损失计算流程
    dot.edge('vla_forward', 'mse_loss')
    dot.edge('mse_loss', 'loss_mean')
    dot.edge('loss_mean', 'backward')
    dot.edge('backward', 'optimizer_step')
    
    return dot

def create_inference_function_call_graph():
    """创建推理函数调用关系图"""
    
    # 创建有向图
    dot = graphviz.Digraph('Inference_Function_Call_Graph', 
                          comment='SmolVLA推理函数调用关系图',
                          format='png')
    
    # 设置图形属性
    dot.attr(rankdir='TB', 
             size='16,20',
             dpi='300',
             fontname='SimHei',
             fontsize='16')
    
    # 设置节点默认属性
    dot.node_attr.update(shape='box',
                         style='filled',
                         fontname='SimHei',
                         fontsize='10',
                         height='0.4',
                         width='2.5')
    
    # 设置边默认属性
    dot.edge_attr.update(fontname='SimHei',
                         fontsize='8',
                         arrowsize='0.8')
    
    # 主推理入口
    with dot.subgraph(name='cluster_main_inference') as main:
        main.attr(label='主推理入口', 
                 style='filled',
                 color='lightblue',
                 fontname='SimHei',
                 fontsize='14')
        
        main.node('select_action', 'SmolVLAPolicy.select_action()\n选择动作', 
                 fillcolor='lightblue',
                 color='darkblue')
        main.node('predict_action_chunk', 'SmolVLAPolicy.predict_action_chunk()\n预测动作块', 
                 fillcolor='lightcyan',
                 color='darkblue')
        main.node('_get_action_chunk', 'SmolVLAPolicy._get_action_chunk()\n获取动作块', 
                 fillcolor='lightcyan',
                 color='darkblue')
    
    # 输入处理
    with dot.subgraph(name='cluster_input_processing') as input_proc:
        input_proc.attr(label='输入处理', 
                       style='filled',
                       color='lightgreen',
                       fontname='SimHei',
                       fontsize='14')
        
        input_proc.node('_prepare_batch_inf', 'SmolVLAPolicy._prepare_batch()\n批次预处理', 
                       fillcolor='lightgreen',
                       color='darkgreen')
        input_proc.node('normalize_inputs_inf', 'Normalize.normalize_inputs()\n输入标准化', 
                       fillcolor='lightgreen',
                       color='darkgreen')
        input_proc.node('populate_queues', 'populate_queues()\n填充队列', 
                       fillcolor='lightgreen',
                       color='darkgreen')
    
    # 特征提取
    with dot.subgraph(name='cluster_feature_extraction_inf') as features:
        features.attr(label='特征提取', 
                     style='filled',
                     color='lightyellow',
                     fontname='SimHei',
                     fontsize='14')
        
        features.node('prepare_images_inf', 'SmolVLAPolicy.prepare_images()\n图像预处理', 
                     fillcolor='lightyellow',
                     color='darkorange')
        features.node('prepare_language_inf', 'SmolVLAPolicy.prepare_language()\n文本token化', 
                     fillcolor='lightyellow',
                     color='darkorange')
        features.node('prepare_state_inf', 'SmolVLAPolicy.prepare_state()\n状态向量化', 
                     fillcolor='lightyellow',
                     color='darkorange')
    
    # 动作采样
    with dot.subgraph(name='cluster_action_sampling') as sampling:
        sampling.attr(label='动作采样', 
                     style='filled',
                     color='lightpink',
                     fontname='SimHei',
                     fontsize='14')
        
        sampling.node('sample_actions', 'VLAFlowMatching.sample_actions()\n采样动作', 
                     fillcolor='lightpink',
                     color='darkred')
        sampling.node('embed_prefix_inf', 'VLAFlowMatching.embed_prefix()\n前缀嵌入', 
                     fillcolor='lightpink',
                     color='darkred')
        sampling.node('kv_cache_compute', 'SmolVLMWithExpertModel.forward()\nKV缓存计算', 
                     fillcolor='lightpink',
                     color='darkred')
        sampling.node('denoise_loop', 'while time >= -dt/2:\n去噪循环', 
                     fillcolor='lightpink',
                     color='darkred')
    
    # 单步去噪
    with dot.subgraph(name='cluster_single_denoise') as denoise:
        denoise.attr(label='单步去噪', 
                    style='filled',
                    color='lightcoral',
                    fontname='SimHei',
                    fontsize='14')
        
        denoise.node('denoise_step', 'VLAFlowMatching.denoise_step()\n去噪步骤', 
                    fillcolor='lightcoral',
                    color='darkred')
        denoise.node('embed_suffix_inf', 'VLAFlowMatching.embed_suffix()\n后缀嵌入', 
                    fillcolor='lightcoral',
                    color='darkred')
        denoise.node('make_att_2d_masks', 'make_att_2d_masks()\n注意力掩码', 
                    fillcolor='lightcoral',
                    color='darkred')
        denoise.node('vlm_forward_inf', 'SmolVLMWithExpertModel.forward()\nVLM前向', 
                    fillcolor='lightcoral',
                    color='darkred')
        denoise.node('euler_step', 'x_t += dt * v_t\ntime += dt\n欧拉步进', 
                    fillcolor='lightcoral',
                    color='darkred')
    
    # 输出处理
    with dot.subgraph(name='cluster_output_processing') as output:
        output.attr(label='输出处理', 
                   style='filled',
                   color='lightsteelblue',
                   fontname='SimHei',
                   fontsize='14')
        
        output.node('unnormalize_outputs', 'Unnormalize.unnormalize_outputs()\n输出反标准化', 
                   fillcolor='lightsteelblue',
                   color='darkblue')
        output.node('_pi_aloha_encode_actions', 'SmolVLAPolicy._pi_aloha_encode_actions()\nALOHA动作编码', 
                   fillcolor='lightsteelblue',
                   color='darkblue')
        output.node('action_queue', 'self._queues[ACTION]\n动作队列管理', 
                   fillcolor='lightsteelblue',
                   color='darkblue')
    
    # 添加边连接
    # 主推理流程
    dot.edge('select_action', 'predict_action_chunk')
    dot.edge('predict_action_chunk', '_get_action_chunk')
    
    # 输入处理流程
    dot.edge('predict_action_chunk', '_prepare_batch_inf')
    dot.edge('_prepare_batch_inf', 'normalize_inputs_inf')
    dot.edge('predict_action_chunk', 'populate_queues')
    
    # 特征提取流程
    dot.edge('_get_action_chunk', 'prepare_images_inf')
    dot.edge('_get_action_chunk', 'prepare_language_inf')
    dot.edge('_get_action_chunk', 'prepare_state_inf')
    
    # 动作采样流程
    dot.edge('_get_action_chunk', 'sample_actions')
    dot.edge('sample_actions', 'embed_prefix_inf')
    dot.edge('sample_actions', 'kv_cache_compute')
    dot.edge('sample_actions', 'denoise_loop')
    
    # 单步去噪流程
    dot.edge('denoise_loop', 'denoise_step')
    dot.edge('denoise_step', 'embed_suffix_inf')
    dot.edge('denoise_step', 'make_att_2d_masks')
    dot.edge('denoise_step', 'vlm_forward_inf')
    dot.edge('denoise_step', 'euler_step')
    dot.edge('euler_step', 'denoise_loop', '循环')
    
    # 输出处理流程
    dot.edge('sample_actions', 'unnormalize_outputs')
    dot.edge('unnormalize_outputs', '_pi_aloha_encode_actions')
    dot.edge('_get_action_chunk', 'action_queue')
    
    return dot

def main():
    """主函数"""
    print("正在生成函数调用关系图...")
    
    # 创建训练函数调用关系图
    train_dot = create_training_function_call_graph()
    train_output_path = './images/training_function_call_graph'
    train_dot.render(train_output_path, cleanup=True)
    print(f"训练函数调用关系图已保存到: {train_output_path}.png")
    
    # 创建推理函数调用关系图
    infer_dot = create_inference_function_call_graph()
    infer_output_path = './images/inference_function_call_graph'
    infer_dot.render(infer_output_path, cleanup=True)
    print(f"推理函数调用关系图已保存到: {infer_output_path}.png")
    
    print("函数调用关系图生成完成！")

if __name__ == "__main__":
    main() 