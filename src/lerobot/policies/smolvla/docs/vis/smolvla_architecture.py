#!/usr/bin/env python3
"""
SmolVLA模型架构可视化脚本
展示SmolVLA的整体架构设计，包括视觉编码器、语言编码器、动作专家和Flow Matching模块
"""

import graphviz

def create_smolvla_architecture():
    """创建SmolVLA模型架构图"""
    
    # 创建有向图
    dot = graphviz.Digraph('SmolVLA_Architecture', 
                          comment='SmolVLA模型架构图',
                          format='png')
    
    # 设置图形属性
    dot.attr(rankdir='TB', 
             size='12,16',
             dpi='300',
             fontname='SimHei',
             fontsize='16')
    
    # 设置节点默认属性
    dot.node_attr.update(shape='box',
                         style='filled',
                         fontname='SimHei',
                         fontsize='14',
                         height='0.8',
                         width='2.0')
    
    # 设置边默认属性
    dot.edge_attr.update(fontname='SimHei',
                         fontsize='12',
                         arrowsize='1.2')
    
    # 输入层
    with dot.subgraph(name='cluster_inputs') as inputs:
        inputs.attr(label='输入层', 
                   style='filled',
                   color='lightblue',
                   fontname='SimHei',
                   fontsize='16')
        
        inputs.node('images', '图像输入\n(512×512)', 
                   fillcolor='lightcyan',
                   color='darkblue')
        inputs.node('language', '语言指令\n(文本token)', 
                   fillcolor='lightcyan',
                   color='darkblue')
        inputs.node('state', '机器人状态\n(关节角度等)', 
                   fillcolor='lightcyan',
                   color='darkblue')
    
    # 视觉编码器
    with dot.subgraph(name='cluster_vision') as vision:
        vision.attr(label='视觉编码器 (SigLIP)', 
                   style='filled',
                   color='lightgreen',
                   fontname='SimHei',
                   fontsize='16')
        
        vision.node('vision_encoder', '视觉特征提取\n(图像→特征向量)', 
                   fillcolor='lightgreen',
                   color='darkgreen')
        vision.node('vision_proj', '视觉投影层\n(特征维度对齐)', 
                   fillcolor='lightgreen',
                   color='darkgreen')
    
    # 语言编码器
    with dot.subgraph(name='cluster_language') as lang:
        lang.attr(label='语言编码器 (SmolVLM2)', 
                 style='filled',
                 color='lightyellow',
                 fontname='SimHei',
                 fontsize='16')
        
        lang.node('text_embed', '文本嵌入层\n(token→向量)', 
                 fillcolor='lightyellow',
                 color='darkorange')
        lang.node('text_encoder', '文本编码器\n(Transformer层)', 
                 fillcolor='lightyellow',
                 color='darkorange')
    
    # 状态处理
    with dot.subgraph(name='cluster_state') as state_proc:
        state_proc.attr(label='状态处理', 
                       style='filled',
                       color='lightpink',
                       fontname='SimHei',
                       fontsize='16')
        
        state_proc.node('state_proj', '状态投影层\n(状态→特征向量)', 
                       fillcolor='lightpink',
                       color='darkred')
    
    # 多模态融合
    with dot.subgraph(name='cluster_fusion') as fusion:
        fusion.attr(label='多模态融合', 
                   style='filled',
                   color='lightcoral',
                   fontname='SimHei',
                   fontsize='16')
        
        fusion.node('concat', '特征拼接\n(图像+文本+状态)', 
                   fillcolor='lightcoral',
                   color='darkred')
        fusion.node('attention', '注意力机制\n(自注意力+交叉注意力)', 
                   fillcolor='lightcoral',
                   color='darkred')
    
    # 动作专家
    with dot.subgraph(name='cluster_expert') as expert:
        expert.attr(label='动作专家网络', 
                   style='filled',
                   color='lightsteelblue',
                   fontname='SimHei',
                   fontsize='16')
        
        expert.node('action_expert', '动作专家\n(Transformer层)', 
                   fillcolor='lightsteelblue',
                   color='darkblue')
        expert.node('action_proj', '动作投影层\n(特征→动作)', 
                   fillcolor='lightsteelblue',
                   color='darkblue')
    
    # Flow Matching
    with dot.subgraph(name='cluster_flow') as flow:
        flow.attr(label='Flow Matching模块', 
                 style='filled',
                 color='lightgoldenrod',
                 fontname='SimHei',
                 fontsize='16')
        
        flow.node('noise', '噪声采样\n(高斯噪声)', 
                 fillcolor='lightgoldenrod',
                 color='darkgoldenrod')
        flow.node('flow_matching', 'Flow Matching\n(噪声→动作)', 
                 fillcolor='lightgoldenrod',
                 color='darkgoldenrod')
        flow.node('denoise', '去噪过程\n(迭代优化)', 
                 fillcolor='lightgoldenrod',
                 color='darkgoldenrod')
    
    # 输出层
    with dot.subgraph(name='cluster_output') as output:
        output.attr(label='输出层', 
                   style='filled',
                   color='lightseagreen',
                   fontname='SimHei',
                   fontsize='16')
        
        output.node('actions', '动作序列\n(关节控制指令)', 
                   fillcolor='lightseagreen',
                   color='darkgreen')
    
    # 添加边连接
    # 输入到编码器
    dot.edge('images', 'vision_encoder', '图像特征提取')
    dot.edge('language', 'text_embed', '文本token化')
    dot.edge('state', 'state_proj', '状态向量化')
    
    # 编码器内部连接
    dot.edge('vision_encoder', 'vision_proj', '特征投影')
    dot.edge('text_embed', 'text_encoder', '文本编码')
    
    # 到融合层
    dot.edge('vision_proj', 'concat', '视觉特征')
    dot.edge('text_encoder', 'concat', '文本特征')
    dot.edge('state_proj', 'concat', '状态特征')
    
    # 融合到专家
    dot.edge('concat', 'attention', '多模态融合')
    dot.edge('attention', 'action_expert', '上下文信息')
    
    # 专家到输出
    dot.edge('action_expert', 'action_proj', '动作特征')
    
    # Flow Matching流程
    dot.edge('noise', 'flow_matching', '初始噪声')
    dot.edge('action_proj', 'flow_matching', '动作预测')
    dot.edge('flow_matching', 'denoise', '去噪步骤')
    dot.edge('denoise', 'actions', '最终动作')
    
    # 添加时间信息
    dot.edge('noise', 'denoise', '时间步信息', style='dashed', color='gray')
    
    return dot

def main():
    """主函数"""
    print("正在生成SmolVLA模型架构图...")
    
    # 创建架构图
    dot = create_smolvla_architecture()
    
    # 保存图片
    output_path = './images/smolvla_architecture'
    dot.render(output_path, cleanup=True)
    
    print(f"架构图已保存到: {output_path}.png")
    print("SmolVLA模型架构图生成完成！")

if __name__ == "__main__":
    main() 