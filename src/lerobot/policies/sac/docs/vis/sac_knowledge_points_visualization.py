#!/usr/bin/env python3
"""
SAC架构知识点和主要技术点可视化脚本
补充生成所有知识点的专门图表
"""

import graphviz

def create_actor_critic_architecture():
    """创建Actor-Critic架构知识点图"""
    dot = graphviz.Digraph(comment='Actor-Critic架构知识点')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='14')
    
    # Actor部分
    with dot.subgraph(name='cluster_actor') as c:
        c.attr(label='Actor (策略网络)', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('actor_input', '状态观测 s', fillcolor='lightcyan')
        c.node('actor_network', '策略网络 π(a|s)', fillcolor='lightblue')
        c.node('actor_output', '动作 a', fillcolor='lightblue')
        c.node('actor_gradient', '策略梯度\n∇θ log π(a|s)', fillcolor='lightblue')
    
    # Critic部分
    with dot.subgraph(name='cluster_critic') as c:
        c.attr(label='Critic (价值网络)', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('critic_input', '状态-动作对 (s,a)', fillcolor='lightyellow')
        c.node('critic_network', '价值网络 Q(s,a)', fillcolor='lightgreen')
        c.node('critic_output', 'Q值', fillcolor='lightgreen')
        c.node('critic_gradient', '价值梯度\n∇Q(s,a)', fillcolor='lightgreen')
    
    # 连接
    dot.edge('actor_input', 'actor_network')
    dot.edge('actor_network', 'actor_output')
    dot.edge('actor_output', 'critic_input')
    dot.edge('critic_input', 'critic_network')
    dot.edge('critic_network', 'critic_output')
    dot.edge('critic_output', 'actor_gradient')
    
    return dot

def create_maximum_entropy_rl():
    """创建最大熵强化学习知识点图"""
    dot = graphviz.Digraph(comment='最大熵强化学习知识点')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 目标函数
    with dot.subgraph(name='cluster_objective') as c:
        c.attr(label='最大熵目标函数', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('objective', 'J(π) = E[∑(r_t + αH(π(·|s_t)))]', fillcolor='lightcyan')
        c.node('entropy_term', '熵正则化项\nαH(π(·|s_t))', fillcolor='lightcyan')
        c.node('reward_term', '奖励项\nr_t', fillcolor='lightcyan')
    
    # 温度参数
    with dot.subgraph(name='cluster_temperature') as c:
        c.attr(label='温度参数控制', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('alpha', '温度参数 α', fillcolor='lightyellow')
        c.node('high_alpha', '高α → 高探索', fillcolor='lightyellow')
        c.node('low_alpha', '低α → 高利用', fillcolor='lightyellow')
    
    # 损失函数
    with dot.subgraph(name='cluster_loss') as c:
        c.attr(label='损失函数', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        c.node('actor_loss', 'L_actor = E[α log π(a|s) - Q(s,a)]', fillcolor='lightpink')
        c.node('temp_loss', 'L_temp = -α(log π(a|s) + H_target)', fillcolor='lightpink')
    
    # 连接
    dot.edge('objective', 'entropy_term')
    dot.edge('objective', 'reward_term')
    dot.edge('entropy_term', 'alpha')
    dot.edge('alpha', 'high_alpha')
    dot.edge('alpha', 'low_alpha')
    dot.edge('alpha', 'actor_loss')
    dot.edge('alpha', 'temp_loss')
    
    return dot

def create_target_network_mechanism():
    """创建目标网络机制知识点图"""
    dot = graphviz.Digraph(comment='目标网络机制知识点')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 主网络
    with dot.subgraph(name='cluster_main') as c:
        c.attr(label='主网络', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('main_network', '主网络 θ', fillcolor='lightcyan')
        c.node('main_update', '频繁更新\n(每步)', fillcolor='lightcyan')
    
    # 目标网络
    with dot.subgraph(name='cluster_target') as c:
        c.attr(label='目标网络', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('target_network', '目标网络 θ_target', fillcolor='lightyellow')
        c.node('target_update', '缓慢更新\n(EMA)', fillcolor='lightyellow')
    
    # 更新公式
    with dot.subgraph(name='cluster_formula') as c:
        c.attr(label='软更新公式', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        c.node('ema_formula', 'θ_target = τ·θ + (1-τ)·θ_target', fillcolor='lightpink')
        c.node('tau_value', 'τ = 0.005 (很小)', fillcolor='lightpink')
    
    # 稳定性
    with dot.subgraph(name='cluster_stability') as c:
        c.attr(label='稳定性优势', style='filled', color='lightgoldenrod', fontname='SimHei', fontsize='16')
        c.node('stable_target', '稳定目标值', fillcolor='lightyellow')
        c.node('reduce_variance', '减少方差', fillcolor='lightyellow')
    
    # 连接
    dot.edge('main_network', 'target_network')
    dot.edge('main_update', 'target_update')
    dot.edge('target_update', 'ema_formula')
    dot.edge('ema_formula', 'tau_value')
    dot.edge('target_network', 'stable_target')
    dot.edge('stable_target', 'reduce_variance')
    
    return dot

def create_ensemble_learning():
    """创建集成学习知识点图"""
    dot = graphviz.Digraph(comment='集成学习知识点')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 多个Critic
    with dot.subgraph(name='cluster_critics') as c:
        c.attr(label='多个Critic网络', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('critic1', 'Q₁(s,a)', fillcolor='lightcyan')
        c.node('critic2', 'Q₂(s,a)', fillcolor='lightcyan')
        c.node('criticN', 'Qₙ(s,a)', fillcolor='lightcyan')
    
    # 最小值操作
    with dot.subgraph(name='cluster_min') as c:
        c.attr(label='最小值操作', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('min_operation', 'Q_min(s,a) = min(Q₁, Q₂, ..., Qₙ)', fillcolor='lightyellow')
        c.node('reduce_overestimation', '减少过估计', fillcolor='lightyellow')
    
    # TD目标
    with dot.subgraph(name='cluster_td') as c:
        c.attr(label='TD目标计算', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        c.node('td_target', 'TD_target = r + γ·Q_min(s\',a\')', fillcolor='lightpink')
        c.node('double_q', '双Q学习思想', fillcolor='lightpink')
    
    # 连接
    dot.edge('critic1', 'min_operation')
    dot.edge('critic2', 'min_operation')
    dot.edge('criticN', 'min_operation')
    dot.edge('min_operation', 'reduce_overestimation')
    dot.edge('min_operation', 'td_target')
    dot.edge('td_target', 'double_q')
    
    return dot

def create_double_q_networks():
    """创建双Q网络技术点图"""
    dot = graphviz.Digraph(comment='双Q网络技术点')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 问题
    with dot.subgraph(name='cluster_problem') as c:
        c.attr(label='过估计问题', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        c.node('overestimation', 'Q值过估计', fillcolor='lightpink')
        c.node('bias', '估计偏差', fillcolor='lightpink')
    
    # 解决方案
    with dot.subgraph(name='cluster_solution') as c:
        c.attr(label='双Q网络解决方案', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('q1', 'Q₁网络\n(选择动作)', fillcolor='lightcyan')
        c.node('q2', 'Q₂网络\n(评估价值)', fillcolor='lightcyan')
    
    # 目标计算
    with dot.subgraph(name='cluster_target') as c:
        c.attr(label='目标计算', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('action_selection', 'a* = argmax Q₁(s\',a)', fillcolor='lightyellow')
        c.node('value_evaluation', 'Q_target = Q₂(s\',a*)', fillcolor='lightyellow')
    
    # 优势
    with dot.subgraph(name='cluster_advantage') as c:
        c.attr(label='技术优势', style='filled', color='lightgoldenrod', fontname='SimHei', fontsize='16')
        c.node('reduce_bias', '减少估计偏差', fillcolor='lightyellow')
        c.node('stable_training', '稳定训练', fillcolor='lightyellow')
    
    # 连接
    dot.edge('overestimation', 'q1')
    dot.edge('bias', 'q2')
    dot.edge('q1', 'action_selection')
    dot.edge('q2', 'value_evaluation')
    dot.edge('action_selection', 'value_evaluation')
    dot.edge('value_evaluation', 'reduce_bias')
    dot.edge('reduce_bias', 'stable_training')
    
    return dot

def create_temperature_auto_tuning():
    """创建温度自动调节技术点图"""
    dot = graphviz.Digraph(comment='温度自动调节技术点')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 温度参数
    with dot.subgraph(name='cluster_temperature') as c:
        c.attr(label='温度参数', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('log_alpha', 'log_α (可学习)', fillcolor='lightcyan')
        c.node('alpha', 'α = exp(log_α)', fillcolor='lightcyan')
    
    # 损失函数
    with dot.subgraph(name='cluster_loss') as c:
        c.attr(label='温度损失', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('temp_loss', 'L(α) = -α(log π(a|s) + H_target)', fillcolor='lightyellow')
        c.node('target_entropy', 'H_target = -|A|/2', fillcolor='lightyellow')
    
    # 自动调节
    with dot.subgraph(name='cluster_auto') as c:
        c.attr(label='自动调节机制', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        c.node('high_entropy', '高熵 → 降低α', fillcolor='lightpink')
        c.node('low_entropy', '低熵 → 提高α', fillcolor='lightpink')
    
    # 平衡
    with dot.subgraph(name='cluster_balance') as c:
        c.attr(label='探索利用平衡', style='filled', color='lightgoldenrod', fontname='SimHei', fontsize='16')
        c.node('exploration', '探索', fillcolor='lightyellow')
        c.node('exploitation', '利用', fillcolor='lightyellow')
    
    # 连接
    dot.edge('log_alpha', 'alpha')
    dot.edge('alpha', 'temp_loss')
    dot.edge('temp_loss', 'target_entropy')
    dot.edge('temp_loss', 'high_entropy')
    dot.edge('temp_loss', 'low_entropy')
    dot.edge('high_entropy', 'exploration')
    dot.edge('low_entropy', 'exploitation')
    
    return dot

def create_tanh_transformation():
    """创建Tanh变换技术点图"""
    dot = graphviz.Digraph(comment='Tanh变换技术点')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 原始分布
    with dot.subgraph(name='cluster_original') as c:
        c.attr(label='原始分布', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('raw_action', '原始动作 a_raw', fillcolor='lightcyan')
        c.node('normal_dist', '正态分布\nN(μ, σ)', fillcolor='lightcyan')
    
    # Tanh变换
    with dot.subgraph(name='cluster_transform') as c:
        c.attr(label='Tanh变换', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('tanh_func', 'a = tanh(a_raw)', fillcolor='lightyellow')
        c.node('bounded_action', '有界动作\n[-1, 1]', fillcolor='lightyellow')
    
    # 概率密度
    with dot.subgraph(name='cluster_density') as c:
        c.attr(label='概率密度变换', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        c.node('density_formula', 'π(a) = π_raw(tanh⁻¹(a))·|det(∂tanh/∂a)|', fillcolor='lightpink')
        c.node('jacobian', '雅可比行列式', fillcolor='lightpink')
    
    # 优势
    with dot.subgraph(name='cluster_advantage') as c:
        c.attr(label='技术优势', style='filled', color='lightgoldenrod', fontname='SimHei', fontsize='16')
        c.node('bounded', '动作有界', fillcolor='lightyellow')
        c.node('differentiable', '可微分', fillcolor='lightyellow')
    
    # 连接
    dot.edge('raw_action', 'tanh_func')
    dot.edge('normal_dist', 'tanh_func')
    dot.edge('tanh_func', 'bounded_action')
    dot.edge('tanh_func', 'density_formula')
    dot.edge('density_formula', 'jacobian')
    dot.edge('bounded_action', 'bounded')
    dot.edge('tanh_func', 'differentiable')
    
    return dot

def create_image_encoder():
    """创建图像编码器技术点图"""
    dot = graphviz.Digraph(comment='图像编码器技术点')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 输入
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='图像输入', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('raw_image', '原始图像\n[H, W, C]', fillcolor='lightcyan')
        c.node('normalize', '标准化', fillcolor='lightcyan')
    
    # 编码器类型
    with dot.subgraph(name='cluster_encoders') as c:
        c.attr(label='编码器类型', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('pretrained', '预训练编码器\n(ResNet)', fillcolor='lightyellow')
        c.node('custom_cnn', '自定义CNN', fillcolor='lightyellow')
    
    # 特征提取
    with dot.subgraph(name='cluster_features') as c:
        c.attr(label='特征提取', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        c.node('feature_maps', '特征图\n[C\', H\', W\']', fillcolor='lightpink')
        c.node('spatial_embed', '空间嵌入', fillcolor='lightpink')
    
    # 输出
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出特征', style='filled', color='lightgoldenrod', fontname='SimHei', fontsize='16')
        c.node('encoded_features', '编码特征\n[latent_dim]', fillcolor='lightyellow')
        c.node('feature_fusion', '特征融合', fillcolor='lightyellow')
    
    # 连接
    dot.edge('raw_image', 'normalize')
    dot.edge('normalize', 'pretrained')
    dot.edge('normalize', 'custom_cnn')
    dot.edge('pretrained', 'feature_maps')
    dot.edge('custom_cnn', 'feature_maps')
    dot.edge('feature_maps', 'spatial_embed')
    dot.edge('spatial_embed', 'encoded_features')
    dot.edge('encoded_features', 'feature_fusion')
    
    return dot

def create_spatial_embedding():
    """创建空间嵌入技术点图"""
    dot = graphviz.Digraph(comment='空间嵌入技术点')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 输入特征
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入特征', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('feature_map', '特征图\n[B, C, H, W]', fillcolor='lightcyan')
    
    # 空间核
    with dot.subgraph(name='cluster_kernel') as c:
        c.attr(label='学习空间核', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('spatial_kernel', '空间核 K\n[C, H, W, F]', fillcolor='lightyellow')
        c.node('learnable', '可学习参数', fillcolor='lightyellow')
    
    # 计算过程
    with dot.subgraph(name='cluster_computation') as c:
        c.attr(label='计算过程', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        c.node('element_wise', '逐元素乘法', fillcolor='lightpink')
        c.node('spatial_reduction', '空间归约\nΣ(h,w)', fillcolor='lightpink')
    
    # 输出
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出嵌入', style='filled', color='lightgoldenrod', fontname='SimHei', fontsize='16')
        c.node('spatial_embed', '空间嵌入\n[B, C×F]', fillcolor='lightyellow')
        c.node('spatial_info', '保留空间信息', fillcolor='lightyellow')
    
    # 连接
    dot.edge('feature_map', 'element_wise')
    dot.edge('spatial_kernel', 'element_wise')
    dot.edge('learnable', 'spatial_kernel')
    dot.edge('element_wise', 'spatial_reduction')
    dot.edge('spatial_reduction', 'spatial_embed')
    dot.edge('spatial_embed', 'spatial_info')
    
    return dot

def create_reward_classifier():
    """创建奖励分类器技术点图"""
    dot = graphviz.Digraph(comment='奖励分类器技术点')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 问题
    with dot.subgraph(name='cluster_problem') as c:
        c.attr(label='稀疏奖励问题', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        c.node('sparse_reward', '稀疏奖励', fillcolor='lightpink')
        c.node('learning_difficulty', '学习困难', fillcolor='lightpink')
    
    # 解决方案
    with dot.subgraph(name='cluster_solution') as c:
        c.attr(label='奖励分类器', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('image_input', '图像输入', fillcolor='lightcyan')
        c.node('classifier', '分类器网络', fillcolor='lightcyan')
        c.node('reward_pred', '奖励预测', fillcolor='lightcyan')
    
    # 损失函数
    with dot.subgraph(name='cluster_loss') as c:
        c.attr(label='分类损失', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('binary_loss', '二元分类\nBCE Loss', fillcolor='lightyellow')
        c.node('multi_loss', '多分类\nCE Loss', fillcolor='lightyellow')
    
    # 优势
    with dot.subgraph(name='cluster_advantage') as c:
        c.attr(label='技术优势', style='filled', color='lightgoldenrod', fontname='SimHei', fontsize='16')
        c.node('dense_reward', '密集奖励', fillcolor='lightyellow')
        c.node('better_learning', '更好学习', fillcolor='lightyellow')
    
    # 连接
    dot.edge('sparse_reward', 'image_input')
    dot.edge('learning_difficulty', 'classifier')
    dot.edge('image_input', 'classifier')
    dot.edge('classifier', 'reward_pred')
    dot.edge('reward_pred', 'binary_loss')
    dot.edge('reward_pred', 'multi_loss')
    dot.edge('binary_loss', 'dense_reward')
    dot.edge('multi_loss', 'better_learning')
    
    return dot

def main():
    """主函数：生成所有知识点和技术点图表"""
    print("正在生成SAC架构知识点和主要技术点可视化图表...")
    
    # 架构知识点图表
    print("1. 生成Actor-Critic架构知识点图...")
    arch_dot = create_actor_critic_architecture()
    arch_dot.render('./images/actor_critic_knowledge', format='png', cleanup=True)
    
    print("2. 生成最大熵强化学习知识点图...")
    entropy_dot = create_maximum_entropy_rl()
    entropy_dot.render('./images/maximum_entropy_knowledge', format='png', cleanup=True)
    
    print("3. 生成目标网络机制知识点图...")
    target_dot = create_target_network_mechanism()
    target_dot.render('./images/target_network_knowledge', format='png', cleanup=True)
    
    print("4. 生成集成学习知识点图...")
    ensemble_dot = create_ensemble_learning()
    ensemble_dot.render('./images/ensemble_learning_knowledge', format='png', cleanup=True)
    
    # 主要技术点图表
    print("5. 生成双Q网络技术点图...")
    doubleq_dot = create_double_q_networks()
    doubleq_dot.render('./images/double_q_networks_tech', format='png', cleanup=True)
    
    print("6. 生成温度自动调节技术点图...")
    temp_dot = create_temperature_auto_tuning()
    temp_dot.render('./images/temperature_auto_tuning_tech', format='png', cleanup=True)
    
    print("7. 生成Tanh变换技术点图...")
    tanh_dot = create_tanh_transformation()
    tanh_dot.render('./images/tanh_transformation_tech', format='png', cleanup=True)
    
    print("8. 生成图像编码器技术点图...")
    img_dot = create_image_encoder()
    img_dot.render('./images/image_encoder_tech', format='png', cleanup=True)
    
    print("9. 生成空间嵌入技术点图...")
    spatial_dot = create_spatial_embedding()
    spatial_dot.render('./images/spatial_embedding_tech', format='png', cleanup=True)
    
    print("10. 生成奖励分类器技术点图...")
    reward_dot = create_reward_classifier()
    reward_dot.render('./images/reward_classifier_tech', format='png', cleanup=True)
    
    print("所有知识点和技术点图表生成完成！")
    print("生成的图片文件：")
    print("- ./images/actor_critic_knowledge.png")
    print("- ./images/maximum_entropy_knowledge.png")
    print("- ./images/target_network_knowledge.png")
    print("- ./images/ensemble_learning_knowledge.png")
    print("- ./images/double_q_networks_tech.png")
    print("- ./images/temperature_auto_tuning_tech.png")
    print("- ./images/tanh_transformation_tech.png")
    print("- ./images/image_encoder_tech.png")
    print("- ./images/spatial_embedding_tech.png")
    print("- ./images/reward_classifier_tech.png")

if __name__ == "__main__":
    main() 