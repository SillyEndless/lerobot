#!/usr/bin/env python3
"""
基于源码的神经网络结构图
展示SmolVLA源码中的实际类结构和方法关系
"""

import graphviz

def create_source_code_structure():
    """创建源码结构图"""
    
    # 创建有向图
    dot = graphviz.Digraph('Source_Code_Structure', 
                          comment='SmolVLA源码结构图',
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
                         width='2.0')
    
    # 设置边默认属性
    dot.edge_attr.update(fontname='SimHei',
                         fontsize='8',
                         arrowsize='0.8')
    
    # SmolVLAPolicy类
    with dot.subgraph(name='cluster_smolvla_policy') as policy:
        policy.attr(label='SmolVLAPolicy (modeling_smolvla.py)', 
                   style='filled',
                   color='lightblue',
                   fontname='SimHei',
                   fontsize='14')
        
        policy.node('SmolVLAPolicy', 'SmolVLAPolicy\n继承自PreTrainedPolicy', 
                   fillcolor='lightblue',
                   color='darkblue')
        
        # 主要方法
        policy.node('__init__', '__init__(config, dataset_stats)', 
                   fillcolor='lightcyan',
                   color='darkblue')
        policy.node('reset', 'reset()', 
                   fillcolor='lightcyan',
                   color='darkblue')
        policy.node('predict_action_chunk', 'predict_action_chunk(batch, noise)', 
                   fillcolor='lightcyan',
                   color='darkblue')
        policy.node('select_action', 'select_action(batch, noise)', 
                   fillcolor='lightcyan',
                   color='darkblue')
        policy.node('forward', 'forward(batch, noise, time)', 
                   fillcolor='lightcyan',
                   color='darkblue')
        
        # 数据准备方法
        policy.node('prepare_images', 'prepare_images(batch)', 
                   fillcolor='lightyellow',
                   color='darkorange')
        policy.node('prepare_language', 'prepare_language(batch)', 
                   fillcolor='lightyellow',
                   color='darkorange')
        policy.node('prepare_state', 'prepare_state(batch)', 
                   fillcolor='lightyellow',
                   color='darkorange')
        policy.node('prepare_action', 'prepare_action(batch)', 
                   fillcolor='lightyellow',
                   color='darkorange')
        
        # 连接
        policy.edge('SmolVLAPolicy', '__init__')
        policy.edge('SmolVLAPolicy', 'reset')
        policy.edge('SmolVLAPolicy', 'predict_action_chunk')
        policy.edge('SmolVLAPolicy', 'select_action')
        policy.edge('SmolVLAPolicy', 'forward')
        policy.edge('SmolVLAPolicy', 'prepare_images')
        policy.edge('SmolVLAPolicy', 'prepare_language')
        policy.edge('SmolVLAPolicy', 'prepare_state')
        policy.edge('SmolVLAPolicy', 'prepare_action')
    
    # VLAFlowMatching类
    with dot.subgraph(name='cluster_vla_flow_matching') as flow:
        flow.attr(label='VLAFlowMatching (modeling_smolvla.py)', 
                 style='filled',
                 color='lightgreen',
                 fontname='SimHei',
                 fontsize='14')
        
        flow.node('VLAFlowMatching', 'VLAFlowMatching\n继承自nn.Module', 
                 fillcolor='lightgreen',
                 color='darkgreen')
        
        # 初始化组件
        flow.node('vlm_with_expert', 'vlm_with_expert\nSmolVLMWithExpertModel', 
                 fillcolor='lightgreen',
                 color='darkgreen')
        flow.node('state_proj', 'state_proj\nLinear(max_state_dim, hidden_size)', 
                 fillcolor='lightgreen',
                 color='darkgreen')
        flow.node('action_in_proj', 'action_in_proj\nLinear(max_action_dim, expert_hidden_size)', 
                 fillcolor='lightgreen',
                 color='darkgreen')
        flow.node('action_out_proj', 'action_out_proj\nLinear(expert_hidden_size, max_action_dim)', 
                 fillcolor='lightgreen',
                 color='darkgreen')
        
        # 主要方法
        flow.node('embed_prefix', 'embed_prefix(images, lang_tokens, state)', 
                 fillcolor='lightseagreen',
                 color='darkgreen')
        flow.node('embed_suffix', 'embed_suffix(noisy_actions, timestep)', 
                 fillcolor='lightseagreen',
                 color='darkgreen')
        flow.node('forward', 'forward(images, lang_tokens, state, actions, noise, time)', 
                 fillcolor='lightseagreen',
                 color='darkgreen')
        flow.node('sample_actions', 'sample_actions(images, lang_tokens, state, noise)', 
                 fillcolor='lightseagreen',
                 color='darkgreen')
        flow.node('denoise_step', 'denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)', 
                 fillcolor='lightseagreen',
                 color='darkgreen')
        
        # 连接
        flow.edge('VLAFlowMatching', 'vlm_with_expert')
        flow.edge('VLAFlowMatching', 'state_proj')
        flow.edge('VLAFlowMatching', 'action_in_proj')
        flow.edge('VLAFlowMatching', 'action_out_proj')
        flow.edge('VLAFlowMatching', 'embed_prefix')
        flow.edge('VLAFlowMatching', 'embed_suffix')
        flow.edge('VLAFlowMatching', 'forward')
        flow.edge('VLAFlowMatching', 'sample_actions')
        flow.edge('VLAFlowMatching', 'denoise_step')
    
    # SmolVLMWithExpertModel类
    with dot.subgraph(name='cluster_smolvlm_expert') as expert:
        expert.attr(label='SmolVLMWithExpertModel (smolvlm_with_expert.py)', 
                   style='filled',
                   color='lightpink',
                   fontname='SimHei',
                   fontsize='14')
        
        expert.node('SmolVLMWithExpertModel', 'SmolVLMWithExpertModel\n继承自nn.Module', 
                   fillcolor='lightpink',
                   color='darkred')
        
        # 组件
        expert.node('vlm', 'vlm\nSmolVLMForConditionalGeneration', 
                   fillcolor='lightpink',
                   color='darkred')
        expert.node('lm_expert', 'lm_expert\nAutoModel (动作专家)', 
                   fillcolor='lightpink',
                   color='darkred')
        expert.node('processor', 'processor\nAutoProcessor', 
                   fillcolor='lightpink',
                   color='darkred')
        
        # 方法
        expert.node('embed_image', 'embed_image(image)', 
                   fillcolor='lightcoral',
                   color='darkred')
        expert.node('embed_language_tokens', 'embed_language_tokens(tokens)', 
                   fillcolor='lightcoral',
                   color='darkred')
        expert.node('forward', 'forward(attention_mask, position_ids, past_key_values, inputs_embeds)', 
                   fillcolor='lightcoral',
                   color='darkred')
        expert.node('forward_attn_layer', 'forward_attn_layer(...)', 
                   fillcolor='lightcoral',
                   color='darkred')
        expert.node('forward_cross_attn_layer', 'forward_cross_attn_layer(...)', 
                   fillcolor='lightcoral',
                   color='darkred')
        
        # 连接
        expert.edge('SmolVLMWithExpertModel', 'vlm')
        expert.edge('SmolVLMWithExpertModel', 'lm_expert')
        expert.edge('SmolVLMWithExpertModel', 'processor')
        expert.edge('SmolVLMWithExpertModel', 'embed_image')
        expert.edge('SmolVLMWithExpertModel', 'embed_language_tokens')
        expert.edge('SmolVLMWithExpertModel', 'forward')
        expert.edge('SmolVLMWithExpertModel', 'forward_attn_layer')
        expert.edge('SmolVLMWithExpertModel', 'forward_cross_attn_layer')
    
    # SmolVLAConfig类
    with dot.subgraph(name='cluster_config') as config:
        config.attr(label='SmolVLAConfig (configuration_smolvla.py)', 
                   style='filled',
                   color='lightgoldenrod',
                   fontname='SimHei',
                   fontsize='14')
        
        config.node('SmolVLAConfig', 'SmolVLAConfig\n继承自PreTrainedConfig', 
                   fillcolor='lightgoldenrod',
                   color='darkgoldenrod')
        
        # 主要配置参数
        config.node('n_obs_steps', 'n_obs_steps: int = 1', 
                   fillcolor='lightyellow',
                   color='darkgoldenrod')
        config.node('chunk_size', 'chunk_size: int = 50', 
                   fillcolor='lightyellow',
                   color='darkgoldenrod')
        config.node('n_action_steps', 'n_action_steps: int = 50', 
                   fillcolor='lightyellow',
                   color='darkgoldenrod')
        config.node('vlm_model_name', 'vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"', 
                   fillcolor='lightyellow',
                   color='darkgoldenrod')
        config.node('num_expert_layers', 'num_expert_layers: int = -1', 
                   fillcolor='lightyellow',
                   color='darkgoldenrod')
        
        # 连接
        config.edge('SmolVLAConfig', 'n_obs_steps')
        config.edge('SmolVLAConfig', 'chunk_size')
        config.edge('SmolVLAConfig', 'n_action_steps')
        config.edge('SmolVLAConfig', 'vlm_model_name')
        config.edge('SmolVLAConfig', 'num_expert_layers')
    
    # 类之间的依赖关系
    dot.edge('SmolVLAPolicy', 'VLAFlowMatching', '包含')
    dot.edge('SmolVLAPolicy', 'SmolVLAConfig', '使用配置')
    dot.edge('VLAFlowMatching', 'SmolVLMWithExpertModel', '包含')
    dot.edge('SmolVLAPolicy', 'SmolVLMWithExpertModel', '通过VLAFlowMatching访问')
    
    # 添加注释
    dot.node('note1', '主要类结构：\n- SmolVLAPolicy: 策略接口\n- VLAFlowMatching: 核心模型\n- SmolVLMWithExpertModel: VLM+专家网络\n- SmolVLAConfig: 配置管理', 
             shape='note',
             fillcolor='lightyellow',
             color='darkorange')
    
    return dot

def main():
    """主函数"""
    print("正在生成源码结构图...")
    
    # 创建源码结构图
    dot = create_source_code_structure()
    
    # 保存图片
    output_path = './images/source_code_structure'
    dot.render(output_path, cleanup=True)
    
    print(f"源码结构图已保存到: {output_path}.png")
    print("源码结构图生成完成！")

if __name__ == "__main__":
    main() 