#!/usr/bin/env python3
"""
训练和推理流程图
展示SmolVLA的训练流程和推理流程
"""

import graphviz

def create_training_flow():
    """创建训练流程图"""
    
    # 创建有向图
    dot = graphviz.Digraph('Training_Flow', 
                          comment='SmolVLA训练流程图',
                          format='png')
    
    # 设置图形属性
    dot.attr(rankdir='TB', 
             size='14,16',
             dpi='300',
             fontname='SimHei',
             fontsize='16')
    
    # 设置节点默认属性
    dot.node_attr.update(shape='box',
                         style='filled',
                         fontname='SimHei',
                         fontsize='11',
                         height='0.6',
                         width='2.0')
    
    # 设置边默认属性
    dot.edge_attr.update(fontname='SimHei',
                         fontsize='10',
                         arrowsize='1.0')
    
    # 数据准备阶段
    with dot.subgraph(name='cluster_data_prep') as data:
        data.attr(label='数据准备阶段', 
                 style='filled',
                 color='lightblue',
                 fontname='SimHei',
                 fontsize='14')
        
        data.node('batch', '批次数据\n(图像+文本+状态+动作)', 
                 fillcolor='lightcyan',
                 color='darkblue')
        data.node('normalize_inputs', 'normalize_inputs(batch)\n输入标准化', 
                 fillcolor='lightcyan',
                 color='darkblue')
        data.node('normalize_targets', 'normalize_targets(batch)\n目标标准化', 
                 fillcolor='lightcyan',
                 color='darkblue')
    
    # 特征提取阶段
    with dot.subgraph(name='cluster_feature_extraction') as features:
        features.attr(label='特征提取阶段', 
                     style='filled',
                     color='lightgreen',
                     fontname='SimHei',
                     fontsize='14')
        
        features.node('prepare_images', 'prepare_images(batch)\n图像预处理', 
                     fillcolor='lightgreen',
                     color='darkgreen')
        features.node('prepare_language', 'prepare_language(batch)\n文本token化', 
                     fillcolor='lightgreen',
                     color='darkgreen')
        features.node('prepare_state', 'prepare_state(batch)\n状态向量化', 
                     fillcolor='lightgreen',
                     color='darkgreen')
        features.node('prepare_action', 'prepare_action(batch)\n动作向量化', 
                     fillcolor='lightgreen',
                     color='darkgreen')
    
    # Flow Matching训练阶段
    with dot.subgraph(name='cluster_flow_training') as flow:
        flow.attr(label='Flow Matching训练阶段', 
                 style='filled',
                 color='lightpink',
                 fontname='SimHei',
                 fontsize='14')
        
        flow.node('sample_noise', 'sample_noise(shape)\n采样噪声ε', 
                 fillcolor='lightpink',
                 color='darkred')
        flow.node('sample_time', 'sample_time(bsize)\n采样时间t', 
                 fillcolor='lightpink',
                 color='darkred')
        flow.node('x_t_calc', 'x_t = t·ε + (1-t)·actions\n线性插值', 
                 fillcolor='lightpink',
                 color='darkred')
        flow.node('u_t_calc', 'u_t = ε - actions\n计算速度场', 
                 fillcolor='lightpink',
                 color='darkred')
    
    # 网络前向传播
    with dot.subgraph(name='cluster_forward') as forward:
        forward.attr(label='网络前向传播', 
                    style='filled',
                    color='lightyellow',
                    fontname='SimHei',
                    fontsize='14')
        
        forward.node('embed_prefix', 'embed_prefix(images, lang, state)\n前缀嵌入', 
                    fillcolor='lightyellow',
                    color='darkorange')
        forward.node('embed_suffix', 'embed_suffix(x_t, time)\n后缀嵌入', 
                    fillcolor='lightyellow',
                    color='darkorange')
        forward.node('vlm_forward', 'vlm_with_expert.forward()\nVLM+专家前向', 
                    fillcolor='lightyellow',
                    color='darkorange')
        forward.node('v_t_pred', 'v_t = action_out_proj(suffix_out)\n预测速度场', 
                    fillcolor='lightyellow',
                    color='darkorange')
    
    # 损失计算
    with dot.subgraph(name='cluster_loss') as loss:
        loss.attr(label='损失计算', 
                 style='filled',
                 color='lightcoral',
                 fontname='SimHei',
                 fontsize='14')
        
        loss.node('mse_loss', 'F.mse_loss(u_t, v_t)\nMSE损失', 
                 fillcolor='lightcoral',
                 color='darkred')
        loss.node('loss_mean', 'loss.mean()\n平均损失', 
                 fillcolor='lightcoral',
                 color='darkred')
        loss.node('backward', 'loss.backward()\n反向传播', 
                 fillcolor='lightcoral',
                 color='darkred')
        loss.node('optimizer_step', 'optimizer.step()\n参数更新', 
                 fillcolor='lightcoral',
                 color='darkred')
    
    # 添加边连接
    # 数据准备流程
    dot.edge('batch', 'normalize_inputs')
    dot.edge('batch', 'normalize_targets')
    
    # 特征提取流程
    dot.edge('batch', 'prepare_images')
    dot.edge('batch', 'prepare_language')
    dot.edge('batch', 'prepare_state')
    dot.edge('batch', 'prepare_action')
    
    # Flow Matching流程
    dot.edge('prepare_action', 'sample_noise')
    dot.edge('sample_noise', 'x_t_calc')
    dot.edge('prepare_action', 'x_t_calc')
    dot.edge('sample_time', 'x_t_calc')
    dot.edge('sample_noise', 'u_t_calc')
    dot.edge('prepare_action', 'u_t_calc')
    
    # 前向传播流程
    dot.edge('prepare_images', 'embed_prefix')
    dot.edge('prepare_language', 'embed_prefix')
    dot.edge('prepare_state', 'embed_prefix')
    dot.edge('x_t_calc', 'embed_suffix')
    dot.edge('sample_time', 'embed_suffix')
    dot.edge('embed_prefix', 'vlm_forward')
    dot.edge('embed_suffix', 'vlm_forward')
    dot.edge('vlm_forward', 'v_t_pred')
    
    # 损失计算流程
    dot.edge('u_t_calc', 'mse_loss')
    dot.edge('v_t_pred', 'mse_loss')
    dot.edge('mse_loss', 'loss_mean')
    dot.edge('loss_mean', 'backward')
    dot.edge('backward', 'optimizer_step')
    
    return dot

def create_inference_flow():
    """创建推理流程图"""
    
    # 创建有向图
    dot = graphviz.Digraph('Inference_Flow', 
                          comment='SmolVLA推理流程图',
                          format='png')
    
    # 设置图形属性
    dot.attr(rankdir='TB', 
             size='14,16',
             dpi='300',
             fontname='SimHei',
             fontsize='16')
    
    # 设置节点默认属性
    dot.node_attr.update(shape='box',
                         style='filled',
                         fontname='SimHei',
                         fontsize='11',
                         height='0.6',
                         width='2.0')
    
    # 设置边默认属性
    dot.edge_attr.update(fontname='SimHei',
                         fontsize='10',
                         arrowsize='1.0')
    
    # 输入处理阶段
    with dot.subgraph(name='cluster_input_processing') as input_proc:
        input_proc.attr(label='输入处理阶段', 
                       style='filled',
                       color='lightblue',
                       fontname='SimHei',
                       fontsize='14')
        
        input_proc.node('obs_batch', '观测批次\n(图像+文本+状态)', 
                       fillcolor='lightcyan',
                       color='darkblue')
        input_proc.node('normalize_inputs_inf', 'normalize_inputs(batch)\n输入标准化', 
                       fillcolor='lightcyan',
                       color='darkblue')
        input_proc.node('prepare_images_inf', 'prepare_images(batch)\n图像预处理', 
                       fillcolor='lightcyan',
                       color='darkblue')
        input_proc.node('prepare_language_inf', 'prepare_language(batch)\n文本token化', 
                       fillcolor='lightcyan',
                       color='darkblue')
        input_proc.node('prepare_state_inf', 'prepare_state(batch)\n状态向量化', 
                       fillcolor='lightcyan',
                       color='darkblue')
    
    # 前缀计算阶段
    with dot.subgraph(name='cluster_prefix_computation') as prefix:
        prefix.attr(label='前缀计算阶段', 
                   style='filled',
                   color='lightgreen',
                   fontname='SimHei',
                   fontsize='14')
        
        prefix.node('embed_prefix_inf', 'embed_prefix(images, lang, state)\n前缀嵌入', 
                   fillcolor='lightgreen',
                   color='darkgreen')
        prefix.node('prefix_attn_masks', 'make_att_2d_masks()\n注意力掩码', 
                   fillcolor='lightgreen',
                   color='darkgreen')
        prefix.node('prefix_position_ids', 'torch.cumsum(pad_masks) - 1\n位置编码', 
                   fillcolor='lightgreen',
                   color='darkgreen')
        prefix.node('kv_cache', 'vlm_with_expert.forward()\n计算KV缓存', 
                   fillcolor='lightgreen',
                   color='darkgreen')
    
    # 去噪采样阶段
    with dot.subgraph(name='cluster_denoising') as denoise:
        denoise.attr(label='去噪采样阶段', 
                    style='filled',
                    color='lightpink',
                    fontname='SimHei',
                    fontsize='14')
        
        denoise.node('init_noise', 'sample_noise(shape)\n初始化噪声', 
                    fillcolor='lightpink',
                    color='darkred')
        denoise.node('init_time', 'time = 1.0\n初始时间', 
                    fillcolor='lightpink',
                    color='darkred')
        denoise.node('denoise_loop', 'while time >= -dt/2:\n去噪循环', 
                    fillcolor='lightpink',
                    color='darkred')
    
    # 单步去噪
    with dot.subgraph(name='cluster_single_step') as step:
        step.attr(label='单步去噪', 
                 style='filled',
                 color='lightyellow',
                 fontname='SimHei',
                 fontsize='14')
        
        step.node('embed_suffix_inf', 'embed_suffix(x_t, timestep)\n后缀嵌入', 
                 fillcolor='lightyellow',
                 color='darkorange')
        step.node('suffix_attn_masks', 'make_att_2d_masks()\n后缀注意力掩码', 
                 fillcolor='lightyellow',
                 color='darkorange')
        step.node('full_attn_masks', 'torch.cat([prefix, suffix])\n完整注意力掩码', 
                 fillcolor='lightyellow',
                 color='darkorange')
        step.node('denoise_forward', 'vlm_with_expert.forward()\n去噪前向', 
                 fillcolor='lightyellow',
                 color='darkorange')
        step.node('v_t_pred_inf', 'v_t = action_out_proj(suffix_out)\n预测速度', 
                 fillcolor='lightyellow',
                 color='darkorange')
        step.node('euler_step', 'x_t += dt * v_t\ntime += dt\n欧拉步进', 
                 fillcolor='lightyellow',
                 color='darkorange')
    
    # 输出处理
    with dot.subgraph(name='cluster_output_processing') as output:
        output.attr(label='输出处理阶段', 
                   style='filled',
                   color='lightcoral',
                   fontname='SimHei',
                   fontsize='14')
        
        output.node('final_actions', 'x_t (最终动作)', 
                   fillcolor='lightcoral',
                   color='darkred')
        output.node('unnormalize', 'unnormalize_outputs(actions)\n输出反标准化', 
                   fillcolor='lightcoral',
                   color='darkred')
        output.node('action_chunk', '动作序列\n(batch_size, n_action_steps, action_dim)', 
                   fillcolor='lightcoral',
                   color='darkred')
    
    # 添加边连接
    # 输入处理流程
    dot.edge('obs_batch', 'normalize_inputs_inf')
    dot.edge('obs_batch', 'prepare_images_inf')
    dot.edge('obs_batch', 'prepare_language_inf')
    dot.edge('obs_batch', 'prepare_state_inf')
    
    # 前缀计算流程
    dot.edge('prepare_images_inf', 'embed_prefix_inf')
    dot.edge('prepare_language_inf', 'embed_prefix_inf')
    dot.edge('prepare_state_inf', 'embed_prefix_inf')
    dot.edge('embed_prefix_inf', 'prefix_attn_masks')
    dot.edge('embed_prefix_inf', 'prefix_position_ids')
    dot.edge('prefix_attn_masks', 'kv_cache')
    dot.edge('prefix_position_ids', 'kv_cache')
    
    # 去噪采样流程
    dot.edge('init_noise', 'denoise_loop')
    dot.edge('init_time', 'denoise_loop')
    dot.edge('denoise_loop', 'embed_suffix_inf')
    
    # 单步去噪流程
    dot.edge('embed_suffix_inf', 'suffix_attn_masks')
    dot.edge('suffix_attn_masks', 'full_attn_masks')
    dot.edge('kv_cache', 'full_attn_masks')
    dot.edge('full_attn_masks', 'denoise_forward')
    dot.edge('denoise_forward', 'v_t_pred_inf')
    dot.edge('v_t_pred_inf', 'euler_step')
    dot.edge('euler_step', 'embed_suffix_inf', '循环')
    
    # 输出处理流程
    dot.edge('euler_step', 'final_actions')
    dot.edge('final_actions', 'unnormalize')
    dot.edge('unnormalize', 'action_chunk')
    
    return dot

def main():
    """主函数"""
    print("正在生成训练和推理流程图...")
    
    # 创建训练流程图
    train_dot = create_training_flow()
    train_output_path = './images/training_flow'
    train_dot.render(train_output_path, cleanup=True)
    print(f"训练流程图已保存到: {train_output_path}.png")
    
    # 创建推理流程图
    infer_dot = create_inference_flow()
    infer_output_path = './images/inference_flow'
    infer_dot.render(infer_output_path, cleanup=True)
    print(f"推理流程图已保存到: {infer_output_path}.png")
    
    print("训练和推理流程图生成完成！")

if __name__ == "__main__":
    main() 