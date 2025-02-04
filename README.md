# Process-the-China-Elderly-Fall-Injury-Dataset-with-Python.
Process the China Elderly Fall Injury Dataset with Python.
以下是手把手带你用Python处理《中国老年人跌倒损伤数据集》的完整流程：

---

### **步骤1：环境准备与数据获取**
#### 1.1 安装必要库
```python
# 在终端执行以下命令
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

#### 1.2 获取Kaggle数据集
1. 注册Kaggle账号（https://www.kaggle.com/）
2. 搜索数据集"China Elderly Fall Injury Dataset"
3. 点击"Download"按钮下载数据集（通常为CSV文件）
4. 解压文件到项目文件夹（假设文件名为`elderly_fall.csv`）

---

### **步骤2：数据加载与初步探索**
```python
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv('elderly_fall.csv')

# 查看数据结构
print(f"数据集维度：{df.shape}")
print("\n前5行数据：")
display(df.head())

# 查看统计摘要
print("\n数据摘要：")
display(df.describe(include='all'))

# 检查缺失值
print("\n缺失值统计：")
display(df.isnull().sum())
```

---

### **步骤3：数据清洗**
#### 3.1 处理缺失值
```python
# 删除缺失超过30%的列
df = df.loc[:, df.isnull().mean() < 0.3]

# 数值型列用中位数填充
num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 类别型列用众数填充
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
```

#### 3.2 异常值处理
```python
# 以年龄字段为例
plt.boxplot(df['Age'])
plt.title('Age Distribution')
plt.show()

# 使用IQR方法处理异常值
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Age'] < (Q1 - 1.5*IQR)) | (df['Age'] > (Q3 + 1.5*IQR)))]
```

---

### **步骤4：特征工程**
#### 4.1 创建新特征
```python
# 示例：创建跌倒风险等级
df['FallRiskLevel'] = pd.cut(df['BalanceTestScore'],
                            bins=[0, 40, 70, 100],
                            labels=['High', 'Medium', 'Low'])

# 时间特征处理（假设有日期字段）
df['InjuryDate'] = pd.to_datetime(df['InjuryDate'])
df['InjuryMonth'] = df['InjuryDate'].dt.month
```

#### 4.2 编码分类变量
```python
from sklearn.preprocessing import LabelEncoder

# 对目标变量编码（假设预测是否骨折）
le = LabelEncoder()
df['Fracture'] = le.fit_transform(df['Fracture'])

# 独热编码
df = pd.get_dummies(df, columns=['Gender', 'ResidenceType'])
```

---

### **步骤5：可视化分析**
#### 5.1 关键指标分布
```python
plt.figure(figsize=(12,6))

# 年龄分布
plt.subplot(1,2,1)
df['Age'].hist(bins=20)
plt.title('Age Distribution')

# 跌倒原因分布
plt.subplot(1,2,2)
df['FallCause'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Fall Causes')

plt.tight_layout()
plt.show()
```

#### 5.2 相关性分析
```python
import seaborn as sns

# 计算相关系数
corr_matrix = df.corr()

# 绘制热力图
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
```

---

### **步骤6：构建预测模型**
#### 6.1 数据拆分
```python
from sklearn.model_selection import train_test_split

X = df.drop(['Fracture', 'InjuryDate'], axis=1)  # 特征
y = df['Fracture']  # 目标变量

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```

#### 6.2 训练随机森林模型
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 初始化模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
print(f"模型准确率：{accuracy_score(y_test, y_pred):.2%}")
```

#### 6.3 特征重要性分析
```python
feature_importance = pd.Series(model.feature_importances_,
                              index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance')
plt.show()
```

---

### **步骤7：输出分析报告**
```python
# 生成描述性统计
report = df.groupby('FallRiskLevel').agg({
    'Age': 'mean',
    'Fracture': 'mean',
    'HospitalizationDays': 'median'
}).reset_index()

print("\n跌倒风险等级分析报告：")
display(report)

# 保存处理后的数据
df.to_csv('processed_fall_data.csv', index=False)
```

---

### **关键要点解析**
1. **数据质量优先**：通过箱线图和IQR方法处理异常值，保证分析结果可靠性
2. **特征工程技巧**：创建`FallRiskLevel`等衍生特征提升模型预测能力
3. **可视化驱动决策**：热力图揭示特征间隐藏关系，指导康复方案设计
4. **模型可解释性**：特征重要性排序帮助识别关键风险因素（如平衡能力、骨密度）

---

### **下一步行动建议**
1. **部署模型**：使用Flask将模型封装为Web服务，供康复师快速评估风险
2. **数据产品化**：将分析结果整合进PPT，向社区医院推介智能评估系统
3. **学术转化**：基于特征重要性结果撰写《老年跌倒生物力学因素分析》论文

```python
# 示例：使用joblib保存模型
import joblib
joblib.dump(model, 'fall_risk_model.pkl')
```

通过这个完整流程，你不仅掌握了Python数据分析技术，更构建了可直接用于专业实践的决策支持工具。接下来可以尝试将分析结果与康复训练方案结合，开发数据驱动的个性化干预计划。
