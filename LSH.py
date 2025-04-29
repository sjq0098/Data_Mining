import numpy as np
import hashlib

def getshingles(text, k=3):  # 增大shingle长度
    text = text.lower().replace(" ", "")
    return {text[i:i+k] for i in range(len(text)-k+1)}

def stable_hash(shingle):
    return int(hashlib.md5(shingle.encode('utf-8')).hexdigest()[:8], 16) % (2**32)

def init_hash_parameters(num_hashes=100):
    np.random.seed(42)
    max_val = 2**31-1
    return [(np.random.randint(1,max_val), np.random.randint(0,max_val)) 
            for _ in range(num_hashes)]

hash_params = init_hash_parameters()

def minhash(shingles, hash_params, prime=998244353):
    return [min((a*stable_hash(s)+b)%prime for s in shingles) 
            for a,b in hash_params]

def lsh_buckets(signatures, bands=25, rows=4):  # 优化LSH参数
    buckets = {}
    for doc_id, sig in signatures.items():
        for b in range(bands):
            band = tuple(sig[b*rows : (b+1)*rows])
            buckets.setdefault((b, band), []).append(doc_id)
    return buckets

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    return intersection / (len(set1 | set2) + 1e-9)  # 避免除零错误

# 包含重复文档的测试数据
documents = {
    "docA": "Global warming is causing the polar ice caps to melt...",
    "docB": "Due to global warming, the polar ice caps are melting...", 
    "docC": "The latest smartphone features...",
    "docD": "The latest smartphone features..."  ,
    "doc1": "The quick brown fox jumps over the lazy dog.",
    "doc2": "The quick brown fox jumps over the lazy dog.",  # 完全相同的文本
    "doc3": "A fast brown canine leaps above a tired hound."  # 语义相同但表述不同
}
# 生成特征
shingles = {doc: getshingles(text,5) for doc,text in documents.items()}
signatures = {doc: minhash(s, hash_params) for doc,s in shingles.items()}

# 执行LSH
buckets = lsh_buckets(signatures, bands=25, rows=4)
candidate_pairs = {tuple(sorted([d1,d2])) 
                  for bucket in buckets.values() 
                  for i,d1 in enumerate(bucket) 
                  for d2 in bucket[i+1:]}

print("Candidate Pairs:")
for pair in candidate_pairs:
    print(f"{pair[0]} vs {pair[1]}: Jaccard = {jaccard_similarity(shingles[pair[0]], shingles[pair[1]]):.2f}")