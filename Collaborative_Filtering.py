import numpy as np
from collections import defaultdict

class HybridCF:
    def __init__(self, alpha=0.5):
        """
        初始化混合协同过滤系统
        alpha: 用户-物品权重平衡参数，范围[0,1]
        alpha=1 时完全使用基于用户的推荐
        alpha=0 时完全使用基于物品的推荐
        """
        self.user_item_matrix = {}  # 用户-物品评分矩阵
        self.item_user_matrix = {}  # 物品-用户评分矩阵
        self.alpha = alpha
        
    def add_rating(self, user_id, item_id, rating):
        """添加评分数据"""
        # 更新用户-物品矩阵
        if user_id not in self.user_item_matrix:
            self.user_item_matrix[user_id] = {}
        self.user_item_matrix[user_id][item_id] = rating
        
        # 更新物品-用户矩阵
        if item_id not in self.item_user_matrix:
            self.item_user_matrix[item_id] = {}
        self.item_user_matrix[item_id][user_id] = rating
        
    def calculate_user_similarity(self, user1, user2):
        """计算用户相似度"""
        common_items = set(self.user_item_matrix[user1].keys()) & \
                      set(self.user_item_matrix[user2].keys())
        
        if not common_items:
            return 0
            
        vector1 = [self.user_item_matrix[user1][item] for item in common_items]
        vector2 = [self.user_item_matrix[user2][item] for item in common_items]
        
        return self._cosine_similarity(vector1, vector2)
        
    def calculate_item_similarity(self, item1, item2):
        """计算物品相似度"""
        common_users = set(self.item_user_matrix[item1].keys()) & \
                      set(self.item_user_matrix[item2].keys())
        
        if not common_users:
            return 0
            
        vector1 = [self.item_user_matrix[item1][user] for user in common_users]
        vector2 = [self.item_user_matrix[item2][user] for user in common_users]
        
        return self._cosine_similarity(vector1, vector2)
        
    def _cosine_similarity(self, vector1, vector2):
        """计算余弦相似度"""
        dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
        norm1 = np.sqrt(sum(v * v for v in vector1))
        norm2 = np.sqrt(sum(v * v for v in vector2))
        
        return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0
        
    def get_user_based_recommendations(self, user_id, n_recommendations=5):
        """基于用户的推荐"""
        if user_id not in self.user_item_matrix:
            return []
            
        # 计算用户相似度
        user_similarities = {}
        for other_user in self.user_item_matrix:
            if other_user != user_id:
                similarity = self.calculate_user_similarity(user_id, other_user)
                user_similarities[other_user] = similarity
                
        # 计算预测评分
        item_scores = defaultdict(float)
        similarity_sums = defaultdict(float)
        
        for other_user, similarity in user_similarities.items():
            if similarity <= 0:
                continue
                
            for item_id, rating in self.user_item_matrix[other_user].items():
                if item_id not in self.user_item_matrix[user_id]:
                    item_scores[item_id] += similarity * rating
                    similarity_sums[item_id] += similarity
                    
        recommendations = []
        for item_id, score in item_scores.items():
            if similarity_sums[item_id] > 0:
                recommendations.append(
                    (item_id, score / similarity_sums[item_id])
                )
                
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:n_recommendations]
        
    def get_item_based_recommendations(self, user_id, n_recommendations=5):
        """基于物品的推荐"""
        if user_id not in self.user_item_matrix:
            return []
            
        # 获取用户没有评分的物品
        rated_items = set(self.user_item_matrix[user_id].keys())
        all_items = set(self.item_user_matrix.keys())
        unrated_items = all_items - rated_items
        
        # 计算预测评分
        predictions = []
        for item_id in unrated_items:
            score = 0
            total_similarity = 0
            
            # 基于用户已评分的物品计算相似度
            for rated_item in rated_items:
                similarity = self.calculate_item_similarity(item_id, rated_item)
                if similarity > 0:
                    score += similarity * self.user_item_matrix[user_id][rated_item]
                    total_similarity += similarity
                    
            if total_similarity > 0:
                predictions.append((item_id, score / total_similarity))
                
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]
        
    def get_hybrid_recommendations(self, user_id, n_recommendations=5):
        """混合推荐"""
        user_based_recs = dict(self.get_user_based_recommendations(user_id, n_recommendations))
        item_based_recs = dict(self.get_item_based_recommendations(user_id, n_recommendations))
        
        # 合并推荐结果
        hybrid_scores = defaultdict(float)
        all_items = set(user_based_recs.keys()) | set(item_based_recs.keys())
        
        for item in all_items:
            user_score = user_based_recs.get(item, 0)
            item_score = item_based_recs.get(item, 0)
            # 使用alpha参数平衡两种方法
            hybrid_scores[item] = self.alpha * user_score + (1 - self.alpha) * item_score
            
        recommendations = [(item, score) for item, score in hybrid_scores.items()]
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:n_recommendations]

# 使用示例
if __name__ == "__main__":
    # 创建混合推荐系统实例
    cf = HybridCF(alpha=0.3)
    
    # 添加示例数据
    cf.add_rating("User1", "ItemA", 5)
    cf.add_rating("User1", "ItemB", 3)
    cf.add_rating("User1", "ItemC", 4)
    
    cf.add_rating("User2", "ItemA", 3)
    cf.add_rating("User2", "ItemB", 4)
    cf.add_rating("User2", "ItemD", 5)
    
    cf.add_rating("User3", "ItemA", 4)
    cf.add_rating("User3", "ItemC", 5)
    cf.add_rating("User3", "ItemD", 2)
    
    # 获取混合推荐结果
    recommendations = cf.get_hybrid_recommendations("User1")
    print("为User1的混合推荐结果:")
    for item_id, score in recommendations:
        print(f"物品: {item_id}, 预测评分: {score:.2f}")