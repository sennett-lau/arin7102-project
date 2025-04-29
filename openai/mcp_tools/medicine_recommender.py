import pandas as pd
import os

def load_webmd_data():
    """
    加载药物数据集
    
    返回:
    DataFrame: 包含药物数据的DataFrame对象
    """
    try:
        # 尝试从工作目录或上级目录读取数据
        if os.path.exists("webmd.csv"):
            df = pd.read_csv("webmd.csv")
        elif os.path.exists("../webmd.csv"):
            df = pd.read_csv("../webmd.csv")
        else:
            # 如果找不到实际数据，使用示例数据
            print("警告：未找到webmd.csv数据文件，使用示例数据代替")
            df = create_sample_data()
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("使用示例数据代替")
        return create_sample_data()

def create_sample_data():
    """
    创建示例数据，用于在无法加载实际数据时使用
    
    返回:
    DataFrame: 包含示例药物数据的DataFrame对象
    """
    sample_data = [
        {
            "Age": "75 or over",
            "Condition": "Stuffy Nose",
            "Date": "9/21/2014",
            "Drug": "25dph-7.5peh",
            "DrugId": 146724,
            "Effectiveness": 5,
            "Satisfaction": 5,
            "Reviews": "I'm a retired physician and of all the meds I have tried for my allergies (seasonal and not) - this one is the most effective for me. When I first began using this drug some years ago - tiredness as a problem but is not currently.",
            "EaseofUse": 5,
            "Sex": "Male",
            "Sides": "Drowsiness, dizziness, dry mouth/nose/throat, headache, upset stomach, constipation, or trouble sleeping may occur.",
            "UsefulCount": 0
        },
        {
            "Age": "25-34",
            "Condition": "Cold Symptoms",
            "Date": "1/13/2011",
            "Drug": "25dph-7.5peh",
            "DrugId": 146724,
            "Effectiveness": 5,
            "Satisfaction": 5,
            "Reviews": "cleared me right up even with my throat hurting it went away after taking the medicine",
            "EaseofUse": 5,
            "Sex": "Female",
            "Sides": "Drowsiness, dizziness, dry mouth/nose/throat, headache, upset stomach, constipation, or trouble sleeping may occur.",
            "UsefulCount": 1
        },
        {
            "Age": "35-44",
            "Condition": "Headache",
            "Date": "5/20/2015",
            "Drug": "Aspirin",
            "DrugId": 120053,
            "Effectiveness": 4,
            "Satisfaction": 4,
            "Reviews": "Helps with my headaches without causing stomach issues.",
            "EaseofUse": 5,
            "Sex": "Female",
            "Sides": "Stomach upset, heartburn, nausea, and allergic reactions may occur.",
            "UsefulCount": 3
        },
        {
            "Age": "45-54",
            "Condition": "Anxiety",
            "Date": "3/15/2016",
            "Drug": "Lorazepam",
            "DrugId": 153421,
            "Effectiveness": 5,
            "Satisfaction": 4,
            "Reviews": "Helps control my anxiety attacks, but makes me drowsy.",
            "EaseofUse": 4,
            "Sex": "Male",
            "Sides": "Drowsiness, dizziness, tiredness, blurred vision, sleep problems, or muscle weakness may occur.",
            "UsefulCount": 8
        },
        {
            "Age": "19-24",
            "Condition": "Acne",
            "Date": "11/02/2017",
            "Drug": "Tretinoin",
            "DrugId": 189756,
            "Effectiveness": 4,
            "Satisfaction": 3,
            "Reviews": "Works well but causes skin dryness and sensitivity to sun.",
            "EaseofUse": 3,
            "Sex": "Female",
            "Sides": "Skin redness, dryness, itching, scaling, mild burning, or worsening of acne may occur during first 2-4 weeks.",
            "UsefulCount": 5
        }
    ]
    return pd.DataFrame(sample_data)

def get_all_conditions(df):
    """
    获取数据集中所有疾病/症状的种类
    
    参数:
    df (DataFrame): 药物数据的DataFrame对象
    
    返回:
    list: 所有疾病/症状的列表
    """
    return sorted(df['Condition'].unique().tolist())

def get_all_age_groups(df):
    """
    获取数据集中所有年龄组
    
    参数:
    df (DataFrame): 药物数据的DataFrame对象
    
    返回:
    list: 所有年龄组的列表
    """
    return sorted(df['Age'].unique().tolist())

def fuzzy_match_condition(df, condition):
    """
    模糊匹配疾病/症状
    
    参数:
    df (DataFrame): 药物数据的DataFrame对象
    condition (str): 用户输入的症状/疾病名称
    
    返回:
    list: 匹配到的条件列表，按匹配度排序
    """
    all_conditions = get_all_conditions(df)
    
    # 完全匹配
    exact_matches = [c for c in all_conditions if c.lower() == condition.lower()]
    if exact_matches:
        return exact_matches
    
    # 包含匹配（输入是条件的子串）
    contains_matches = [c for c in all_conditions if condition.lower() in c.lower()]
    if contains_matches:
        return contains_matches
    
    # 关键词匹配（条件中的词是输入的子串）
    keyword_matches = [c for c in all_conditions if any(kw.lower() in condition.lower() for kw in c.split())]
    if keyword_matches:
        return keyword_matches
    
    # 如果以上都没有匹配，尝试更宽松的匹配
    # 例如，检查输入中的每个词是否出现在条件中
    loose_matches = []
    for c in all_conditions:
        # 将输入和条件分解为词
        condition_words = condition.lower().split()
        c_words = c.lower().split()
        
        # 计算有多少个词匹配
        matching_words = sum(1 for word in condition_words if any(word in c_word for c_word in c_words))
        
        if matching_words > 0:
            loose_matches.append((c, matching_words))
    
    # 按匹配词数排序
    loose_matches.sort(key=lambda x: x[1], reverse=True)
    
    if loose_matches:
        return [match[0] for match in loose_matches]
    
    return []

def recommend_medicine(df, age, condition, sex):
    """
    根据年龄、症状和性别推荐药物及其副作用
    
    参数:
    df (DataFrame): 药物数据的DataFrame对象
    age (str): 年龄段，例如："25-34", "75 or over"
    condition (str): 症状，例如："Stuffy Nose", "Cold Symptoms"
    sex (str): 性别，例如："Male", "Female"
    
    返回:
    dict: 包含推荐药物名称和可能副作用的字典
    """
    # 模糊匹配条件
    if condition not in get_all_conditions(df):
        matched_conditions = fuzzy_match_condition(df, condition)
        if matched_conditions:
            print(f"未找到精确匹配的症状'{condition}'，使用模糊匹配结果: {matched_conditions[0]}")
            print(f"其他可能的匹配: {', '.join(matched_conditions[1:5]) if len(matched_conditions) > 1 else '无'}")
            condition = matched_conditions[0]
        else:
            return {"message": f"无法找到与'{condition}'相关的症状。可用疾病列表: {', '.join(get_all_conditions(df)[:10])}..."}
    
    # 筛选符合条件的药物
    filtered_df = df[(df['Age'] == age) & (df['Condition'] == condition) & (df['Sex'] == sex)]
    
    if filtered_df.empty:
        # 如果没有完全匹配的，放宽条件尝试只匹配症状
        filtered_df = df[df['Condition'] == condition]
        if not filtered_df.empty:
            print(f"未找到完全匹配的记录，忽略年龄和性别限制")
        
        if filtered_df.empty:
            return {"message": f"没有找到治疗'{condition}'的药物推荐。"}
    
    # 按满意度和有效性排序选取最佳药物
    best_match = filtered_df.sort_values(by=['Satisfaction', 'Effectiveness'], ascending=False).iloc[0]
    
    # 查找相同药物的所有评论，可能提供更多信息
    same_drug_reviews = df[df['Drug'] == best_match['Drug']]
    review_count = len(same_drug_reviews)
    avg_satisfaction = same_drug_reviews['Satisfaction'].mean()
    
    return {
        "drug_name": best_match['Drug'],
        "side_effects": best_match['Sides'],
        "effectiveness": int(best_match['Effectiveness']),
        "satisfaction": int(best_match['Satisfaction']),
        "reviews": best_match['Reviews'],
        "similar_cases": review_count,
        "average_satisfaction": str(round(avg_satisfaction, 1))
    }

def get_multiple_recommendations(df, age, condition, sex, limit=3):
    """
    获取多个药物推荐
    
    参数:
    df (DataFrame): 药物数据的DataFrame对象
    age (str): 年龄段，例如："25-34", "75 or over"
    condition (str): 症状，例如："Stuffy Nose", "Cold Symptoms"
    sex (str): 性别，例如："Male", "Female"
    limit (int): 返回的推荐数量
    
    返回:
    list: 包含多个药物推荐的列表
    """
    # 模糊匹配条件
    if condition not in get_all_conditions(df):
        matched_conditions = fuzzy_match_condition(df, condition)
        if matched_conditions:
            print(f"未找到精确匹配的症状'{condition}'，使用模糊匹配结果: {matched_conditions[0]}")
            print(f"其他可能的匹配: {', '.join(matched_conditions[1:5]) if len(matched_conditions) > 1 else '无'}")
            condition = matched_conditions[0]
        else:
            return [{"message": f"无法找到与'{condition}'相关的症状。可用疾病列表: {', '.join(get_all_conditions(df)[:10])}..."}]
    
    # 筛选符合条件的药物
    filtered_df = df[(df['Age'] == age) & (df['Condition'] == condition) & (df['Sex'] == sex)]
    
    if filtered_df.empty:
        # 如果没有完全匹配的，放宽条件尝试只匹配症状
        filtered_df = df[df['Condition'] == condition]
        if not filtered_df.empty:
            print(f"未找到完全匹配的记录，忽略年龄和性别限制")
        
        if filtered_df.empty:
            return [{"message": f"没有找到治疗'{condition}'的药物推荐。"}]
    
    # 按满意度和有效性排序选取最佳药物
    top_matches = filtered_df.sort_values(by=['Satisfaction', 'Effectiveness'], ascending=False).head(limit)
    
    recommendations = []
    for _, match in top_matches.iterrows():
        # 查找相同药物的所有评论
        same_drug_reviews = df[df['Drug'] == match['Drug']]
        review_count = len(same_drug_reviews)
        avg_satisfaction = same_drug_reviews['Satisfaction'].mean()
        
        recommendations.append({
            "drug_name": match['Drug'],
            "side_effects": match['Sides'],
            "effectiveness": match['Effectiveness'],
            "satisfaction": match['Satisfaction'],
            "reviews": match['Reviews'],
            "similar_cases": review_count,
            "average_satisfaction": round(avg_satisfaction, 1)
        })
    
    return recommendations

def print_recommendation(rec, index=None):
    """
    打印单个药物推荐
    
    参数:
    rec (dict): 药物推荐信息
    index (int, optional): 推荐的索引
    """
    prefix = f"推荐 #{index+1}: " if index is not None else ""
    
    if "message" in rec:
        print(f"{prefix}{rec['message']}")
        return
    
    print(f"{prefix}{rec['drug_name']} (满意度: {rec['satisfaction']}/5, 有效性: {rec['effectiveness']}/5)")
    print(f"  副作用: {rec['side_effects']}")
    print(f"  评价样本: {rec['reviews'][:100]}..." if len(rec['reviews']) > 100 else f"  评价样本: {rec['reviews']}")
    print(f"  类似案例数量: {rec['similar_cases']}, 平均满意度: {rec['average_satisfaction']}/5")

# 使用示例
if __name__ == "__main__":
    print("药物推荐系统")
    print("=" * 50)
    
    # 加载数据
    data = load_webmd_data()
    
    # 获取并显示所有疾病类型
    all_conditions = get_all_conditions(data)
    print(f"数据中包含的疾病/症状类型 ({len(all_conditions)}个):")
    print(", ".join(all_conditions))
    print()
    
    # 获取并显示所有年龄组
    all_age_groups = get_all_age_groups(data)
    print(f"数据中包含的年龄组 ({len(all_age_groups)}个):")
    print(", ".join(all_age_groups))
    print("=" * 50)
    
    # 交互式测试
    print("交互式药物推荐系统 (按Ctrl+C退出)")
    try:
        while True:
            print("\n请输入以下信息进行药物推荐:")
            user_age = input("年龄组 (例如: 25-34): ")
            if not user_age:
                user_age = all_age_groups[0]  # 使用默认值
                print(f"使用默认年龄组: {user_age}")
                
            user_condition = input("症状/疾病: ")
            if not user_condition:
                continue
                
            user_sex = input("性别 (Male/Female): ")
            if not user_sex:
                user_sex = "Female"  # 使用默认值
                print(f"使用默认性别: {user_sex}")
            
            num_recommendations = input("需要推荐数量 (默认1): ")
            try:
                num_recommendations = int(num_recommendations) if num_recommendations else 1
            except ValueError:
                num_recommendations = 1
                print("无效的数量，使用默认值1")
            
            print("\n正在查找推荐...")
            if num_recommendations > 1:
                results = get_multiple_recommendations(data, user_age, user_condition, user_sex, num_recommendations)
                print(f"\n为'{user_condition}'找到 {len(results)} 个药物推荐:")
                print("-" * 50)
                for i, result in enumerate(results):
                    print_recommendation(result, i)
                    if i < len(results) - 1:
                        print("-" * 40)
            else:
                result = recommend_medicine(data, user_age, user_condition, user_sex)
                print(f"\n为'{user_condition}'找到以下推荐:")
                print("-" * 50)
                print_recommendation(result)
            
            again = input("\n是否继续查询? (y/n): ")
            if again.lower() != 'y':
                print("\n感谢使用药物推荐系统!")
                break
    except KeyboardInterrupt:
        print("\n感谢使用药物推荐系统!") 