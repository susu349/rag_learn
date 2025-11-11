import pandas as pd
from pathlib import Path

# 定义文件路径
base_dir = Path(r'D:\医疗机器人')
q_path = base_dir / 'question' / 'question.csv'
a_path = base_dir / 'answer' / 'answer.csv'
output_path = base_dir / 'merged_df.csv'

# 读取CSV文件
print("正在读取问题文件...")
questions_df = pd.read_csv(q_path)
print(f"问题数据: {len(questions_df)} 条记录")

print("正在读取答案文件...")
answers_df = pd.read_csv(a_path)
print(f"答案数据: {len(answers_df)} 条记录")

# 合并问题和答案数据集（按question_id合并）
print("正在按question_id合并数据...")
merged_df = pd.merge(
    questions_df,
    answers_df,
    on='question_id',  # 按question_id进行合并
    how='inner',  # 只保留两个数据集都有的question_id
    suffixes=('_question', '_answer')
)

print(f"\n合并统计信息:")
print(f"  合并后总记录数: {len(merged_df)} 条")
print(f"  唯一question_id数: {merged_df['question_id'].nunique()} 个")
print(f"  平均每个问题有答案数: {len(merged_df) / merged_df['question_id'].nunique():.2f} 个")

# 查看合并后的数据
print("\n合并后的数据预览:")
print(merged_df.head(10))
print(f"\n数据列名: {merged_df.columns.tolist()}")

# 保存合并后的数据
print(f"\n正在保存到: {output_path}")
merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print("保存完成！")