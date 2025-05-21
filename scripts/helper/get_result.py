import json

# 初始化变量
total_generate_time = 0.0
total_tokens_per_second = 0.0
count = 0
min_generate_time = float('inf')  # 初始化为正无穷大
min_tokens_per_second = float('inf')
max_tokens_per_second=-float('inf')

# 打开 JSONL 文件
with open('/home/lg5/project/vlas/llava/serve.jsonl', 'r') as file:
    for line in file:
        # 解析每一行的 JSON 数据
        data = json.loads(line)
        
        # 获取当前值
        current_gt = data['generate_time']
        current_tps = data['tokens_per_second']
        
        # 累加总和
        total_generate_time += current_gt
        total_tokens_per_second += current_tps
        
        # 更新最小值
        if current_gt < min_generate_time:
            min_generate_time = current_gt
        if current_tps < min_tokens_per_second:
            min_tokens_per_second = current_tps
        if current_tps > max_tokens_per_second:
           max_tokens_per_second = current_tps
        
        # 计数
        count += 1

# 计算平均值
average_generate_time = total_generate_time / count
average_tokens_per_second = total_tokens_per_second / count

# 输出结果
print(f"Minimum generate_time: {min_generate_time:.4f}")
print(f"Average generate_time: {average_generate_time:.4f}")
print(f"Minimum tokens_per_second: {min_tokens_per_second:.2f}")
print(f"Average tokens_per_second: {average_tokens_per_second:.2f}")
print(f"maximum tokens_per_second: {max_tokens_per_second:.2f}")