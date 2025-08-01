#!/usr/bin/env python3
"""
Flow Matching技术可视化脚本
展示SmolVLA中Flow Matching的工作原理，从噪声到动作的去噪过程
"""

import graphviz

def create_flow_matching_diagram():
    """创建Flow Matching技术图"""
    
    # 创建有向图
    dot = graphviz.Digraph('Flow_Matching_Process', 
                          comment='Flow Matching去噪过程',
                          format='png')
    
    # 设置图形属性
    dot.attr(rankdir='LR', 
             size='16,10',
             dpi='300',
             fontname='SimHei',
             fontsize='16')
    
    # 设置节点默认属性
    dot.node_attr.update(shape='box',
                         style='filled',
                         fontname='SimHei',
                         fontsize='12',
                         height='0.6',
                         width='1.8')
    
    # 设置边默认属性
    dot.edge_attr.update(fontname='SimHei',
                         fontsize='10',
                         arrowsize='1.0')
    
    # 时间轴
    with dot.subgraph(name='cluster_timeline') as timeline:
        timeline.attr(label='时间轴 (t: 1.0 → 0.0)', 
                     style='filled',
                     color='lightgray',
                     fontname='SimHei',
                     fontsize='14')
        
        timeline.node('t1', 't=1.0\n(纯噪声)', 
                     fillcolor='lightcoral',
                     color='darkred')
        timeline.node('t2', 't=0.8\n(部分去噪)', 
                     fillcolor='lightyellow',
                     color='darkorange')
        timeline.node('t3', 't=0.5\n(中等去噪)', 
                     fillcolor='lightgreen',
                     color='darkgreen')
        timeline.node('t4', 't=0.2\n(接近目标)', 
                     fillcolor='lightblue',
                     color='darkblue')
        timeline.node('t5', 't=0.0\n(目标动作)', 
                     fillcolor='lightseagreen',
                     color='darkgreen')
    
    # 数学公式
    with dot.subgraph(name='cluster_math') as math:
        math.attr(label='数学公式', 
                 style='filled',
                 color='lightcyan',
                 fontname='SimHei',
                 fontsize='14')
        
        math.node('x_t', 'x_t = t·ε + (1-t)·x_0\n(线性插值)', 
                 fillcolor='lightcyan',
                 color='darkblue')
        math.node('u_t', 'u_t = ε - x_0\n(速度场)', 
                 fillcolor='lightcyan',
                 color='darkblue')
        math.node('loss', 'L = ||v_θ(x_t,t) - u_t||²\n(损失函数)', 
                 fillcolor='lightcyan',
                 color='darkblue')
    
    # 网络预测
    with dot.subgraph(name='cluster_network') as network:
        network.attr(label='神经网络预测', 
                    style='filled',
                    color='lightpink',
                    fontname='SimHei',
                    fontsize='14')
        
        network.node('v_theta', 'v_θ(x_t, t)\n(速度场预测)', 
                    fillcolor='lightpink',
                    color='darkred')
        network.node('denoise_step', 'x_{t-1} = x_t + dt·v_θ\n(欧拉步进)', 
                    fillcolor='lightpink',
                    color='darkred')
    
    # 添加边连接
    # 时间轴连接
    dot.edge('t1', 't2', '去噪步骤1')
    dot.edge('t2', 't3', '去噪步骤2')
    dot.edge('t3', 't4', '去噪步骤3')
    dot.edge('t4', 't5', '去噪步骤4')
    
    # 数学公式连接
    dot.edge('t1', 'x_t', '噪声ε')
    dot.edge('t5', 'x_t', '目标x_0')
    dot.edge('x_t', 'u_t', '计算速度场')
    dot.edge('u_t', 'loss', '真实速度场')
    
    # 网络预测连接
    dot.edge('x_t', 'v_theta', '输入(x_t, t)')
    dot.edge('v_theta', 'loss', '预测速度场')
    dot.edge('v_theta', 'denoise_step', '预测速度')
    dot.edge('denoise_step', 't2', '更新x_t')
    
    # 添加注释
    dot.node('note1', 'Flow Matching核心思想：\n学习从噪声到目标的连续路径', 
             shape='note',
             fillcolor='lightyellow',
             color='darkorange')
    dot.node('note2', '训练时：学习速度场\n推理时：沿速度场积分', 
             shape='note',
             fillcolor='lightyellow',
             color='darkorange')
    
    dot.edge('note1', 'x_t', style='dashed', color='gray')
    dot.edge('note2', 'v_theta', style='dashed', color='gray')
    
    return dot

def main():
    """主函数"""
    print("正在生成Flow Matching技术图...")
    
    # 创建Flow Matching图
    dot = create_flow_matching_diagram()
    
    # 保存图片
    output_path = './images/flow_matching_process'
    dot.render(output_path, cleanup=True)
    
    print(f"Flow Matching图已保存到: {output_path}.png")
    print("Flow Matching技术图生成完成！")

if __name__ == "__main__":
    main() 