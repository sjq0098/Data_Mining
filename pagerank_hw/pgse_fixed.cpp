 #include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <queue>
#include <tuple>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <thread>

namespace fs = std::filesystem;

// —— 全局配置 —— 
const double beta       = 0.85;     // 跳转概率
const double epsilon    = 1e-6;     // 收敛阈值
const int    max_iter   = 100;      // 最大迭代次数
const int    block_size = 1000;     // 每个条带大小
const std::string data_file = "Data.txt";

// 安全删除文件的辅助函数，带有重试逻辑
bool safe_remove_file(const std::string& filename, int max_retries = 3) {
    for (int i = 0; i < max_retries; ++i) {
        try {
            if (fs::exists(filename)) {
                fs::remove(filename);
                return true;
            }
            return true; // 文件不存在也算成功
        } catch (const fs::filesystem_error& e) {
            std::cerr << "删除文件失败: " << e.what() << "，尝试重试 (" << i + 1 << "/" << max_retries << ")" << std::endl;
            // 等待一段时间再重试
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    return false;
}

// 安全重命名文件的辅助函数，带有重试逻辑
bool safe_rename_file(const std::string& old_name, const std::string& new_name, int max_retries = 3) {
    for (int i = 0; i < max_retries; ++i) {
        try {
            // 如果目标文件存在，先尝试删除
            if (fs::exists(new_name)) {
                fs::remove(new_name);
            }
            fs::rename(old_name, new_name);
            return true;
        } catch (const fs::filesystem_error& e) {
            std::cerr << "重命名文件失败: " << e.what() << "，尝试重试 (" << i + 1 << "/" << max_retries << ")" << std::endl;
            // 等待一段时间再重试
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    return false;
}

// —— 第1步：读取边文件，统计 outdeg 并分条带 —— 
void create_edge_stripes(int N, int num_stripes) {
    // 创建临时目录用于保存条带文件
    std::string stripe_dir = "stripe_files";
    if (!fs::exists(stripe_dir)) {
        fs::create_directory(stripe_dir);
    }
    
    // 打开所有条带输出流
    std::vector<std::ofstream> stripe_fs;
    stripe_fs.reserve(num_stripes);
    for(int i = 0; i < num_stripes; ++i) {
        stripe_fs.emplace_back(stripe_dir + "/stripe_" + std::to_string(i) + ".txt");
    }
    // 读入边，写到对应条带
    std::ifstream fin(data_file);
    int u, v;
    while(fin >> u >> v) {
        int sid = v / block_size;
        if (sid >= 0 && sid < num_stripes) {
            stripe_fs[sid] << u << " " << v << "\n";
        }
    }
    for(auto &ofs : stripe_fs) ofs.close();
}

// —— 第2步：初始化 PageRank 向量，按条带写磁盘 —— 
void write_initial_r(int N, int num_stripes) {
    // 创建临时目录用于保存R向量文件
    std::string r_dir = "r_files";
    if (!fs::exists(r_dir)) {
        fs::create_directory(r_dir);
    }
    
    for(int sid = 0; sid < num_stripes; ++sid) {
        int start = sid * block_size;
        int end   = std::min((sid+1)*block_size, N);
        std::ofstream rf(r_dir + "/r_stripe_" + std::to_string(sid) + ".txt");
        for(int i = start; i < end; ++i) {
            rf << (1.0 / N) << "\n";
        }
    }
}

// —— 第3步：外存条带 PageRank 迭代 —— 
void external_stripe_pagerank(int N, int num_stripes,
                              const std::unordered_map<int,int> &outdeg) 
{
    std::string r_dir = "r_files";
    std::string stripe_dir = "stripe_files";
    
    for(int iter = 0; iter < max_iter; ++iter) {
        // 3.1 计算死节点泄漏量
        double leaked = 0.0;
        for(int sid = 0; sid < num_stripes; ++sid) {
            int start = sid * block_size;
            std::ifstream rf(r_dir + "/r_stripe_" + std::to_string(sid) + ".txt");
            std::string line;
            int idx = 0;
            while(std::getline(rf, line)) {
                double val = std::stod(line);
                int node = start + idx;
                if (outdeg.find(node) == outdeg.end()) {
                    leaked += beta * val;
                }
                ++idx;
            }
        }
        double leaked_share = leaked / N;
        double base = (1 - beta) / N;

        double diff = 0.0;
        // 3.2 按目的条带 vid 更新
        for(int vid = 0; vid < num_stripes; ++vid) {
            int vstart = vid * block_size;
            int vend   = std::min((vid+1)*block_size, N);
            int m      = vend - vstart;

            // 新条带缓冲
            std::vector<double> r_new(m, base + leaked_share);

            // 枚举所有源条带 uid
            for(int uid = 0; uid < num_stripes; ++uid) {
                int ustart = uid * block_size;
                int uend   = std::min((uid+1)*block_size, N);

                // 读旧 r_stripe_uid
                std::vector<double> r_block;
                {
                    std::string r_file = r_dir + "/r_stripe_" + std::to_string(uid) + ".txt";
                    std::ifstream uf(r_file);
                    if (!uf) {
                        std::cerr << "无法打开文件: " << r_file << std::endl;
                        continue;
                    }
                    std::string l;
                    while(std::getline(uf,l)) {
                        r_block.push_back(std::stod(l));
                    }
                }
                
                // 读对应目的边 stripe_vid
                std::string edge_file = stripe_dir + "/stripe_" + std::to_string(vid) + ".txt";
                std::ifstream ef(edge_file);
                if (!ef) {
                    // 如果边文件不存在，跳过
                    continue;
                }
                
                int uu, vv;
                while(ef >> uu >> vv) {
                    if (uu >= ustart && uu < uend) {
                        auto it = outdeg.find(uu);
                        if (it != outdeg.end()) {
                            int d = it->second;
                            int local_u = uu - ustart;
                            int local_v = vv - vstart;
                            if (local_u >= 0 && local_u < r_block.size() &&
                                local_v >= 0 && local_v < r_new.size()) {
                                r_new[local_v] += beta * (r_block[local_u] / d);
                            }
                        }
                    }
                }
            }

            // 写入临时新条带并累加 diff
            std::string new_name = r_dir + "/r_new_stripe_" + std::to_string(vid) + ".txt";
            std::ofstream wf(new_name);
            for(double val : r_new) {
                wf << val << "\n";
            }
            wf.close(); // 确保文件写入完成并关闭
            
            // 计算 L1 差分
            {
                std::string old_name = r_dir + "/r_stripe_" + std::to_string(vid) + ".txt";
                std::ifstream oldf(old_name);
                std::ifstream newf(new_name);
                if (!oldf || !newf) {
                    std::cerr << "无法打开文件进行比较: " << old_name << " 或 " << new_name << std::endl;
                    continue;
                }
                std::string lo, ln;
                while(std::getline(oldf,lo) && std::getline(newf,ln)) {
                    diff += std::fabs(std::stod(ln) - std::stod(lo));
                }
            }
        }

        // 3.3 用新文件替换旧条带
        for(int sid = 0; sid < num_stripes; ++sid) {
            std::string old_file = r_dir + "/r_new_stripe_" + std::to_string(sid) + ".txt";
            std::string new_file = r_dir + "/r_stripe_" + std::to_string(sid) + ".txt";
            
            // 使用安全重命名函数
            if (!safe_rename_file(old_file, new_file)) {
                std::cerr << "警告: 无法重命名文件 " << old_file << " 到 " << new_file << std::endl;
                // 备选方案: 复制新内容到旧文件，然后删除新文件
                std::ifstream src(old_file, std::ios::binary);
                std::ofstream dst(new_file, std::ios::binary | std::ios::trunc);
                if (src && dst) {
                    dst << src.rdbuf();
                    src.close();
                    dst.close();
                    safe_remove_file(old_file);
                }
            }
        }

        std::cout << "Iter " << iter << " diff=" << diff << std::endl;
        if (diff < epsilon) {
            std::cout << "Converged.\n";
            break;
        }
    }
}

// —— 第4步：k 路归并输出 Top-K —— 
void merge_top100(int N, int num_stripes, int K=100) {
    std::string r_dir = "r_files";
    std::string result_dir = "results";
    
    // 创建结果目录
    if (!fs::exists(result_dir)) {
        fs::create_directory(result_dir);
    }
    
    // 4.1 对每个条带内部排序并写入 sorted_sid.txt
    std::vector<std::string> sorted_files;
    for(int sid = 0; sid < num_stripes; ++sid) {
        int start = sid * block_size;
        std::vector<std::pair<double,int>> buf;
        std::string r_file = r_dir + "/r_stripe_" + std::to_string(sid) + ".txt";
        std::ifstream rf(r_file);
        if (!rf) {
            std::cerr << "无法打开文件: " << r_file << std::endl;
            continue;
        }
        
        std::string line;
        int idx = 0;
        while(std::getline(rf,line)) {
            buf.emplace_back(std::stod(line), start + idx);
            ++idx;
        }
        rf.close(); // 确保文件关闭
        
        std::sort(buf.begin(), buf.end(),
                  [](auto &a, auto &b){ return a.first > b.first; });
        
        std::string tmp = result_dir + "/sorted_" + std::to_string(sid) + ".txt";
        std::ofstream of(tmp);
        for(auto &p : buf) {
            of << p.second << "\t" << p.first << "\n";
        }
        of.close(); // 确保文件关闭
        sorted_files.push_back(tmp);
    }

    // 4.2 k 路归并
    struct Item { double neg_score; int sid, node; };
    auto cmp = [](Item const &a, Item const &b){
        return a.neg_score > b.neg_score;
    };
    std::priority_queue<Item,std::vector<Item>,decltype(cmp)> heap(cmp);

    std::vector<std::ifstream> ifs;
    for(auto &fn : sorted_files) {
        ifs.emplace_back(fn);
    }
    // 初始化堆
    for(int sid = 0; sid < ifs.size(); ++sid) {
        std::string line;
        if (std::getline(ifs[sid], line)) {
            std::istringstream ss(line);
            int nid; double score;
            ss >> nid >> score;
            heap.push({-score, sid, nid});
        }
    }

    // 输出 Top-K
    std::ofstream out(result_dir + "/res_stripe_ext.txt");
    for(int i = 0; i < K && !heap.empty(); ++i) {
        auto it = heap.top(); heap.pop();
        out << it.node << "\t" << -it.neg_score << "\n";
        // 读取该文件下一行
        std::string line;
        if (std::getline(ifs[it.sid], line)) {
            std::istringstream ss(line);
            int nid; double score;
            ss >> nid >> score;
            heap.push({-score, it.sid, nid});
        }
    }
    out.close();

    // 关闭所有文件流
    for(auto &f : ifs) {
        f.close();
    }

    // 清理临时文件
    for(auto &fn: sorted_files) {
        if (!safe_remove_file(fn)) {
            std::cerr << "警告: 无法删除临时文件 " << fn << std::endl;
        }
    }
}

// —— 主流程 —— 
int main(){
    // 统计 outdeg & N
    std::unordered_map<int,int> outdeg;
    std::unordered_set<int> nodes;
    {
        std::ifstream fin(data_file);
        int u,v;
        while(fin >> u >> v){
            nodes.insert(u); nodes.insert(v);
            outdeg[u]++;
        }
    }
    int N = *std::max_element(nodes.begin(), nodes.end()) + 1;
    int num_stripes = std::ceil(double(N)/block_size);
    std::cout << "Total nodes = " << N
              << ", stripes = " << num_stripes << std::endl;

    create_edge_stripes(N, num_stripes);
    write_initial_r(N, num_stripes);
    external_stripe_pagerank(N, num_stripes, outdeg);
    merge_top100(N, num_stripes, 100);

    std::cout << "Done. Results in results/res_stripe_ext.txt\n";
    return 0;
}
 